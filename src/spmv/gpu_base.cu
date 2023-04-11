// Copyright 2023 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph_gpu.h"
#include "timer.h"
#include "spmv_util.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
typedef float ValueT;

// spmv_csr_warp: CSR SpMV kernels based on a warp model (one warp per row).
//   Each row of the CSR matrix is assigned to a warp. The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with 
//   the x vector, in parallel. This division of work implies that the
//   CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned). On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of 
//   work. Since an entire 32-thread warp is assigned to each row, many 
//   threads will remain idle when their row contains a small number of
//   elements. This code relies on implicit synchronization among threads
//   in a warp. Note that the texture cache is used for accessing the x vector.

__global__ void spmv_warp(int m, const eidType* Ap, 
                          const vidType* Aj, const ValueT* Ax, 
                          const ValueT* x, ValueT* y) {
  __shared__ ValueT sdata[BLOCK_SIZE + 16];                 // padded to avoid reduction ifs
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
    ValueT sum = 0;
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

void SpmvSolver(GraphF &g, const ValueT *h_x, ValueT *h_y) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_Ap = g.in_rowptr();
  auto h_Aj = g.in_colidx();	
  auto h_Ax = g.get_elabel_ptr();
  eidType *d_Ap;
  vidType *d_Aj;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(eidType)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(eidType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(vidType), cudaMemcpyHostToDevice));

  ValueT *d_Ax, *d_x, *d_y;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
  CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));

  size_t nthreads = BLOCK_SIZE;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int nSM = deviceProp.multiProcessorCount;
  auto max_blocks_per_SM = maximum_residency(spmv_warp, nthreads, 0);
  size_t max_blocks = max_blocks_per_SM * nSM;
  size_t nblocks = std::min(max_blocks, DIVIDE_INTO(size_t(m), size_t(WARPS_PER_BLOCK)));
  printf("CUDA SpMV solver (%ld CTAs, %ld threads/CTA) ...\n", nblocks, nthreads);

  Timer t;
  t.Start();
  spmv_warp<<<nblocks, nthreads>>>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);   
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  //CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
  t.Stop();

  double time = t.Millisecs();
  float gbyte = bytes_per_spmv(m, nnz);
  float GFLOPs = (time == 0) ? 0 : (2 * nnz / time) / 1e6;
  float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
  printf("runtime = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", time, GFLOPs, GBYTEs);
  //ValueT *y2 = (ValueT *)malloc(m * sizeof(ValueT));
  //for (int i = 0; i < m; i ++) y2[i] = h_y[i];
  //SpmvSerial(m, nnz, h_Ap, h_Aj, h_Ax, h_x, y2);
  //double error = l2_error(m, y2, h_y);
  //printf("L2 error %f\n", error);
  //free(y2);

  CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, m * sizeof(ValueT), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_Ap));
  CUDA_SAFE_CALL(cudaFree(d_Aj));
  CUDA_SAFE_CALL(cudaFree(d_Ax));
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_y));
}

