#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <stdint.h>
#include <algorithm>
#include "ctimer.h"
#include "common.h"
#include <iostream>

typedef float T;
typedef uint32_t vidType;
typedef int64_t eidType;

__global__ void spmv_warp(int m, const eidType* Ap, 
                          const vidType* Aj, const T* Ax, 
                          const T* x, T* y) {
  __shared__ T sdata[BLOCK_SIZE + 16];                 // padded to avoid reduction ifs
  __shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
  sdata[threadIdx.x + 16] = 0.0;
  __syncthreads();

  int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for(int row = warp_id; row < m; row += num_warps) {
    // use two threads to fetch Ap[row] and Ap[row+1]
    // this is considerably faster than the straightforward version
    if(thread_lane < 2)
      ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
    const int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
    const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

    // compute local sum
    T sum = 0;
    for(int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE)
      //sum += Ax[offset] * x[Aj[offset]];
      sum += Ax[offset] * __ldg(x + Aj[offset]);

    // reduce local sums to row sum (ASSUME: warpsize 32)
    sdata[threadIdx.x] = sum; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

    // first thread writes warp result
    if (thread_lane == 0) y[row] += sdata[threadIdx.x];
  }
}

extern "C"
void SpmvSolver(size_t m, size_t nnz, const eidType *h_Ap, const vidType *h_Aj, const T *h_Ax, const T *h_x, T *h_y) {
  eidType *d_Ap;
  vidType *d_Aj;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(eidType)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(eidType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(vidType), cudaMemcpyHostToDevice));

  T *d_Ax, *d_x, *d_y;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(T) * nnz));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(T) * m));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(T) * m));
  CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(T), cudaMemcpyHostToDevice));

  size_t nthreads = BLOCK_SIZE;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int nSM = deviceProp.multiProcessorCount;
  auto max_blocks_per_SM = maximum_residency(spmv_warp, nthreads, 0);
  size_t max_blocks = max_blocks_per_SM * nSM;
  size_t nblocks = std::min(max_blocks, DIVIDE_INTO(size_t(m), size_t(WARPS_PER_BLOCK)));
  printf("CUDA SpMV solver (%ld CTAs, %ld threads/CTA) ...\n", nblocks, nthreads);
 
  ctimer_t t;
  ctimer_start(&t);

  spmv_warp<<<nblocks, nthreads>>>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
 
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "SpMV-cuda-kernel");

  CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, m * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_Ap));
  CUDA_SAFE_CALL(cudaFree(d_Aj));
  CUDA_SAFE_CALL(cudaFree(d_Ax));
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_y));
}

