#include <iostream>

#define CHECK_ERROR(errorMessage) {                                    \
  cudaError_t err = cudaGetLastError();                                    \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
	errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    exit(EXIT_FAILURE);                                                  \
  }                                                                        \
}
#define TILE_N 16 
#define TILE_TB_HEIGHT 8
#define TILE_M (TILE_N*TILE_TB_HEIGHT)
typedef float DType;

__global__ void sgemm_kernel(const float *A, int lda,
                             const float *B, int ldb,
                                   float* C, int ldc, 
                             int k, float alpha, float beta) {
  // Partial results 
  float c[TILE_N];
  for (int i=0; i < TILE_N; i++)
    c[i] = 0.0f;
  int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
  int m = blockIdx.x * TILE_M + mid;
  int n = blockIdx.y * TILE_N + threadIdx.x;
  __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];
  for (int i = 0; i < k; i+=TILE_TB_HEIGHT) {
    float a; 
    b_s[threadIdx.y][threadIdx.x]=B[n + (i+threadIdx.y)*ldb];
    __syncthreads();
    for (int j = 0; j < TILE_TB_HEIGHT; j++) {
      a = A[m + (i+j)*lda];
      for (int kk = 0; kk < TILE_N; kk++)
        c[kk] += a * b_s[j][kk];

    }
    __syncthreads();
  }
  int t = ldc*blockIdx.y * TILE_N + m;
  for (int i = 0; i < TILE_N; i++) {
    C[t+i*ldc] = C[t+i*ldc] * beta + alpha * c[i];
  }
}

void sgemm(char transa, char transb,
           int m, int n, int k,
           float alpha, const float *A, int lda,
           const float *B, int ldb, float beta,
           float *C, int ldc ) {
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  // In this code we assume the matrix sizes are multiple of tile size
  if ((m%TILE_M) || (n%TILE_N)) {
    std::cerr << "unsupported size of matrix. m should be multiple of " << TILE_M
      << "; n should be multiple of " << TILE_N << std::endl;
  }

  DType *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(DType)*m*k);
  cudaMalloc(&d_B, sizeof(DType)*k*n);
  cudaMalloc(&d_C, sizeof(DType)*m*n);
  cudaMemcpy(d_A, A, sizeof(DType)*m*k, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(DType)*k*n, cudaMemcpyHostToDevice);

  dim3 grid( m/TILE_M, n/TILE_N ), threads( TILE_N, TILE_TB_HEIGHT );
  sgemm_kernel<<<grid, threads>>>( A, lda, B, ldb, C, ldc, k, alpha, beta);
  cudaDeviceSynchronize();
  CHECK_ERROR("mySgemm");

  cudaMemcpy(C, d_C, m*n*sizeof(DType), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

