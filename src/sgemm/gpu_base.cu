#include <iostream>

#define CHECK_ERROR(errorMessage) {                                    \
  cudaError_t err = cudaGetLastError();                                    \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
	errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    exit(EXIT_FAILURE);                                                  \
  }                                                                        \
}

// Parameters of tile sizes
#define TILE_SZ 16 
typedef float DType;

__global__ void sgemm_kernel(const DType *A, int lda, const DType *B, int ldb,
                             DType* C, int ldc, int k, DType alpha, DType beta ) {
  DType c = 0.0f;
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = 0; i < k; ++i) {
    DType a = A[m + i * lda]; 
    DType b = B[n + i * ldb];
    c += a * b;
  }
  C[m+n*ldc] = C[m+n*ldc] * beta + alpha * c;
}

void sgemm(char transa, char transb, 
           int m, int n, int k, DType alpha, 
           const DType *A, int lda, const DType *B, int ldb,
           DType beta, DType *C, int ldc ) {
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  // In this code we assume the matrix sizes are multiple of tile size
  if ((m%TILE_SZ) || (n%TILE_SZ)) {
    std::cerr << "unsupported size of matrix. m should be multiple of " << TILE_SZ
      << "; n should be multiple of " << TILE_SZ << std::endl;
  }

  DType *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(DType)*m*k);
  cudaMalloc(&d_B, sizeof(DType)*k*n);
  cudaMalloc(&d_C, sizeof(DType)*m*n);
  cudaMemcpy(d_A, A, sizeof(DType)*m*k, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(DType)*k*n, cudaMemcpyHostToDevice);

  dim3 grid(m/TILE_SZ, n/TILE_SZ);
  dim3 threads(TILE_SZ, TILE_SZ);
  sgemm_kernel<<<grid, threads>>>(d_A, lda, d_B, ldb, d_C, ldc, k, alpha, beta);
  cudaDeviceSynchronize();
  CHECK_ERROR("mySgemm");

  cudaMemcpy(C, d_C, m*n*sizeof(DType), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

