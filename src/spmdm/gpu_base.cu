#include "ctimer.h"
#include <stdint.h>
typedef float DType;
typedef int64_t vidType;
typedef uint32_t eidType;


// SPMM with CSR.
// C = SpMM(SpA, B) + C
// BT is transposed B
__global__ void SpMMCsrKernel(vidType m, eidType nnz, int n,
                              const eidType* __restrict__ Ap,
                              const vidType* __restrict__ Aj,
                              const DType* __restrict__ Ax,
                              const DType* __restrict__ BT,
                              DType* __restrict__ C) {
  int tx = blockIdx.y * blockDim.x + threadIdx.x;
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < m) {
    while (tx < n) {
      DType sum = 0;
      for (int off = Ap[ty]; off < Ap[ty+1]; off++) {
        vidType k = Aj[off]; // column id
        DType value = Ax[off];   // A[i][k]
        //sum += value * B[k*n + tx]; // A[i][k] * B[k][j]
        sum += value * BT[tx*m + k]; // A[i][k] * BT[j][k]
      }
      C[ty * n + tx] += sum;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// A: m x m 
// B: m x n
// BT:n x m
// C: m x n
extern "C"
void SpmDm(char transa, char transb, 
           vidType m, eidType nnz, int n,
           DType alpha, const eidType *Ap,
           const vidType *Aj, const DType *Ax, 
           int lda, const DType *BT, int ldb, 
           DType beta, DType *C, int ldc) {
  printf("CUDA SpMDM solver\n");
  ctimer_t t;
  ctimer_start(&t);
  eidType * d_Ap;
  vidType * d_Aj;
  DType *d_Ax, *d_BT, *d_C;

  cudaMalloc(&d_Ap, sizeof(eidType)*(m+1));
  cudaMalloc(&d_Aj, sizeof(vidType)*nnz);
  cudaMalloc(&d_Ax, sizeof(DType)*nnz);
  cudaMalloc(&d_BT, sizeof(DType)*m*n);
  cudaMalloc(&d_C, sizeof(DType)*m*n);
  cudaMemcpy(d_Ap, Ap, sizeof(eidType)*(m+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Aj, Aj, sizeof(vidType)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ax, Ax, sizeof(DType)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_BT, BT, sizeof(DType)*m*n, cudaMemcpyHostToDevice);
  dim3 blockSize(32, 32);
  int nrows = (int)ceil((float)m/32);
  int ncols = (int)ceil((float)n/32);
  dim3 gridSize(nrows, ncols);

  SpMMCsrKernel<<<gridSize, blockSize>>>(m, nnz, n, d_Ap, d_Aj, d_Ax, d_BT, d_C);

  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "SpmDm");
  //float gbyte = bytes_per_spmdm(m, nnz) / 10e9;
  //float GFLOPs = 2*nnz / time / 10e9;
  //float GBYTEs = gbyte / time;
  //printf("Throughput: compute %5.2f GFLOP/s, memory %5.1f GB/s\n", GFLOPs, GBYTEs);
  cudaMemcpy(C, d_C, m*n*sizeof(DType), cudaMemcpyDeviceToHost);
  cudaFree(d_Ap);
  cudaFree(d_Aj);
  cudaFree(d_Ax);
  cudaFree(d_BT);
  cudaFree(d_C);
}

