
typedef unsigned InTy;
typedef unsigned OutTy;
typedef unsigned size_type;

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

void scanLargeDeviceArray(OutTy *d_out, InTy *d_in, int length);
void scanSmallDeviceArray(OutTy *d_out, InTy *d_in, int length);
void scanLargeEvenDeviceArray(OutTy *d_out, InTy *d_in, int length);

int nextPowerOfTwo(int x) {
  int power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

extern "C"
void prefix_sum(size_type length, const InTy* in, OutTy *out) {
  if (length == 0) return;
  InTy* d_in;
  OutTy* d_out;
  cudaMalloc((void**) &d_in, length * sizeof(InTy));
  cudaMalloc((void**) &d_out, length * sizeof(OutTy));
  cudaMemcpy(d_in, in, length * sizeof(InTy), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, length * sizeof(OutTy), cudaMemcpyHostToDevice);

  // start timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  size_t nthreads = 512;
  size_t nblocks = (length - 1) / nthreads + 1;
  int powerOfTwo = nextPowerOfTwo(length);
  if (length > ELEMENTS_PER_BLOCK) {
    scanLargeDeviceArray(d_out, d_in, length);
  } else {
    scanSmallDeviceArray(d_out, d_in, length);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsedTime = 0;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaMemcpy(out, d_out, length * sizeof(OutTy), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  cudaFree(d_in);
}

__global__ void add(OutTy *output, int length, InTy *n1, OutTy *n2) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * length;
  output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

__global__ void add(OutTy *output, int length, OutTy *n) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * length;
  output[blockOffset + threadID] += n[blockID];
}

__global__ void prescan(OutTy *output, InTy *input, int n, int powerOfTwo) {
  extern __shared__ int temp[];// allocated on invocation
  int threadID = threadIdx.x;
  if (threadID < n) {
    temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
    temp[2 * threadID + 1] = input[2 * threadID + 1];
  } else {
    temp[2 * threadID] = 0;
    temp[2 * threadID + 1] = 0;
  }
  int offset = 1;
  for (int d = powerOfTwo >> 1; d > 0; d >>= 1) { // build sum in place up the tree
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element
  for (int d = 1; d < powerOfTwo; d *= 2) { // traverse down tree & build scan
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  if (threadID < n) {
    output[2 * threadID] = temp[2 * threadID]; // write results to device memory
    output[2 * threadID + 1] = temp[2 * threadID + 1];
  }
}

__global__ void prescan_large(OutTy *output, InTy *input, int n, OutTy *sums) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * n;
  extern __shared__ int temp[];
  temp[2 * threadID] = input[blockOffset + (2 * threadID)];
  temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];
  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) { // build sum in place up the tree
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  __syncthreads();
  if (threadID == 0) {
    sums[blockID] = temp[n - 1];
    temp[n - 1] = 0;
  }
  for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      auto t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  output[blockOffset + (2 * threadID)] = temp[2 * threadID];
  output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}

void scanLargeDeviceArray(OutTy *d_out, InTy *d_in, int length) {
  int remainder = length % (ELEMENTS_PER_BLOCK);
  if (remainder == 0) {
    scanLargeEvenDeviceArray(d_out, d_in, length);
  } else {
    // perform a large scan on a compatible multiple of elements
    int lengthMultiple = length - remainder;
    scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple);
    // scan the remaining elements and add the (inclusive) last element of the large scan to this
    auto startOfOutputArray = &(d_out[lengthMultiple]);
    scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);
    add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
  }
}

void scanSmallDeviceArray(OutTy *d_out, InTy *d_in, int length) {
  int powerOfTwo = nextPowerOfTwo(length);
  prescan<< <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> >(d_out, d_in, length, powerOfTwo);
}

void scanLargeEvenDeviceArray(OutTy *d_out, InTy *d_in, int length) {
  const int blocks = length / ELEMENTS_PER_BLOCK;
  const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);
  OutTy *d_sums, *d_incr;
  cudaMalloc((void **)&d_sums, blocks * sizeof(int));
  cudaMalloc((void **)&d_incr, blocks * sizeof(int));
  prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
  const int sumsArrThreadsNeeded = (blocks + 1) / 2;
  if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
    // perform a large scan on the sums arr
    scanLargeDeviceArray(d_incr, d_sums, blocks);
  } else {
    // only need one block to scan sums arr so can use small scan
    scanSmallDeviceArray(d_incr, d_sums, blocks);
  }
  add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);
  cudaFree(d_sums);
  cudaFree(d_incr);
}

