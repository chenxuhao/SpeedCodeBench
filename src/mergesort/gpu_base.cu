#include <iostream>

#define checkCudaErrors( call) do {										\
   cudaError err = call;												\
   if( cudaSuccess != err) {											\
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
               __FILE__, __LINE__, cudaGetErrorString( err) );			\
   exit(EXIT_FAILURE);													\
   } } while (0)

typedef int DataT;
__global__ void gpu_mergesort(DataT*, DataT*, DataT, DataT, DataT, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(DataT*, DataT*, DataT, DataT, DataT);

#define min(a, b) (a < b ? a : b)

extern "C"
void MergeSort(DataT* data, DataT size) {
  dim3 threadsPerBlock;
  dim3 blocksPerGrid;
  threadsPerBlock.x = 32;
  threadsPerBlock.y = 1;
  threadsPerBlock.z = 1;
  blocksPerGrid.x = 8;
  blocksPerGrid.y = 1;
  blocksPerGrid.z = 1;

  // Allocate two arrays on the GPU
  // we switch back and forth between them during the sort
  DataT* D_data;
  DataT* D_swp;
  dim3* D_threads;
  dim3* D_blocks;
  // Actually allocate the two arrays
  checkCudaErrors(cudaMalloc((void**) &D_data, size * sizeof(DataT)));
  checkCudaErrors(cudaMalloc((void**) &D_swp, size * sizeof(DataT)));
  // Copy from our input list into the first array
  checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(DataT), cudaMemcpyHostToDevice));
  // Copy the thread / block info to the GPU as well
  checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
  checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));
  checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));
  DataT* A = D_data;
  DataT* B = D_swp;
  DataT nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;
  // Slice up the list and give pieces of it to each thread, letting the pieces grow
  // bigger and bigger until the whole list is sorted
  for (int width = 2; width < (size << 1); width <<= 1) {
    DataT slices = size / ((nThreads) * width) + 1;
    // Actually call the kernel
    gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);
    // Switch the input / output arrays instead of copying them around
    A = A == D_data ? D_swp : D_data;
    B = B == D_data ? D_swp : D_data;
  }
  // Get the list back from the GPU
  checkCudaErrors(cudaMemcpy(data, A, size * sizeof(DataT), cudaMemcpyDeviceToHost));
  // Free the GPU memory
  checkCudaErrors(cudaFree(A));
  checkCudaErrors(cudaFree(B));
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
  int x;
  return threadIdx.x +
         threadIdx.y * (x  = threads->x) +
         threadIdx.z * (x *= threads->y) +
         blockIdx.x  * (x *= threads->z) +
         blockIdx.y  * (x *= blocks->z) +
         blockIdx.z  * (x *= blocks->y);
}

// Perform a full mergesort on our section of the data.
__global__ void gpu_mergesort(DataT* source, DataT* dest, DataT size, DataT width, DataT slices, dim3* threads, dim3* blocks) {
  unsigned int idx = getIdx(threads, blocks);
  DataT start = width*idx*slices, middle, end;
  for (DataT slice = 0; slice < slices; slice++) {
    if (start >= size)
      break;
    middle = min(start + (width >> 1), size);
    end = min(start + width, size);
    gpu_bottomUpMerge(source, dest, start, middle, end);
    start += width;
  }
}

// Finally, sort something
// gets called by gpu_mergesort() for each slice
__device__ void gpu_bottomUpMerge(DataT* source, DataT* dest, DataT start, DataT middle, DataT end) {
  DataT i = start;
  DataT j = middle;
  for (DataT k = start; k < end; k++) {
    if (i < middle && (j >= end || source[i] < source[j])) {
      dest[k] = source[i];
      i++;
    } else {
      dest[k] = source[j];
      j++;
    }
  }
}

