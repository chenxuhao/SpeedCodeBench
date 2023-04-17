#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>

#define checkCudaErrors( call) do {										\
   cudaError err = call;												\
   if( cudaSuccess != err) {											\
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
               __FILE__, __LINE__, cudaGetErrorString( err) );			\
   exit(EXIT_FAILURE);													\
   } } while (0)

// This version uses contiguous threads, but its interleaved addressing results in many shared memory bank conflicts.
template <class T>
__global__ void reduce1(T *g_idata, T *g_odata, unsigned int n) {
  extern __shared__ T sdata[];
  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? g_idata[i] : 0;
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s=1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// This version uses sequential addressing -- no divergence or bank conflicts.
template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
  extern __shared__ T sdata[];
  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? g_idata[i] : 0;
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads) {
  threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
  blocks = (n + threads - 1) / threads;
}

// Wrapper function for kernel launch
template <class T>
void reduce(int size, int threads, int blocks, T *d_idata, T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
  reduce2<T><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
}

typedef int T;
extern "C"
void reduction(int n, T *h_idata, T &sum, T &max_num, T &min_num) {
  assert(n > 0);
  unsigned int bytes = n * sizeof(T);
  int cpuFinalThreshold = 1;
  int numThreads = 512;
  int numBlocks = (n-1)/numThreads + 1;
  if (numBlocks == 1) cpuFinalThreshold = 1;

  T *h_odata = (T *) malloc(numBlocks*sizeof(T));
  T *d_idata = NULL;
  T *d_odata = NULL;
  checkCudaErrors(cudaMalloc((void **) &d_idata, bytes));
  checkCudaErrors(cudaMalloc((void **) &d_odata, numBlocks*sizeof(T)));
  // copy data directly to device memory
  checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks*sizeof(T), cudaMemcpyHostToDevice));

  bool cpuFinalReduction = false;
  bool needReadBack = true;
  T gpu_result = 0;
  reduce<T>(n, numThreads, numBlocks, d_idata, d_odata);
  int blocks = 0, threads = 0, maxBlocks = 0, maxThreads = 0;
  if (cpuFinalReduction) {
    // sum partial sums from each block on CPU
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(T), cudaMemcpyDeviceToHost));
    for (int i=0; i<numBlocks; i++) {
      gpu_result += h_odata[i];
    }
    needReadBack = false;
  } else {
    int s = numBlocks;
    while (s > cpuFinalThreshold) {
      int threads = 0, blocks = 0;
      getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
      reduce<T>(s, threads, blocks, d_odata, d_odata);
      s = (s + threads - 1) / threads;
      //s = (s + (threads*2-1)) / (threads*2);
    }
    if (s > 1) {
      // copy result from device to host
      checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));
      for (int i=0; i < s; i++) {
        gpu_result += h_odata[i];
      }
      needReadBack = false;
    }
  }
  cudaDeviceSynchronize();
  if (needReadBack)
    checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
  sum = gpu_result;
}

