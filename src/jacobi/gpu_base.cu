# include <stdio.h>
# include "ctimer.h"

#define checkCudaErrors( call) do {										\
   cudaError err = call;												\
   if( cudaSuccess != err) {											\
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
               __FILE__, __LINE__, cudaGetErrorString( err) );			\
   exit(EXIT_FAILURE);													\
   } } while (0)


typedef double DataTy;

__global__ void jacobi_kernel(DataTy* A, DataTy* b, DataTy* x_old, DataTy* x_new, int Ni, int Nj) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < Ni) {
    //int idx = i*Nj;
    //DataTy sigma = 0.0;
    //for (int j=0; j<Nj; j++)
    //  if (i != j) sigma += A[idx_Ai + j] * x_old[j];
    //x_new[i] = (b[i] - sigma) / A[idx_Ai + i];

    DataTy sigma = b[i];
    for (int j = 0; j < Nj; j++ )
      if ( j != i ) sigma -= A[i+j*Ni] * x_old[j];
    x_new[i] = sigma / A[i+i*Ni];
  }
}

// The linear system A*x=b is to be solved.
// Parameters:
//    Input,  int N, the order of the matrix.
//    Input,  double A[N,N], the matrix.
//    Input,  double B[N], the right hand side.
//    Input,  double X[N], the current solution estimate.
//    Output, double X[N], the solution estimate updated by
extern "C"
void jacobi(int iter, int n, double *A, double *b, double *x0, double *x1) {
  ctimer_t t;
  ctimer_start(&t);
  int Ni = n;
  int Nj = n;
  int N = Ni * Nj;

  // Allocate memory on the device
  DataTy *d_x0, *d_x1, *d_A, *d_b;
  checkCudaErrors(cudaMalloc((void **) &d_A,  N *sizeof(DataTy)));
  checkCudaErrors(cudaMalloc((void **) &d_b,  Ni*sizeof(DataTy)));
  checkCudaErrors(cudaMalloc((void **) &d_x0, Ni*sizeof(DataTy)));
  checkCudaErrors(cudaMalloc((void **) &d_x1, Ni*sizeof(DataTy)));

  // Copy data -> device
  cudaMemcpy(d_A,  A,  sizeof(DataTy)*N,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,  b,  sizeof(DataTy)*Ni, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x0, x0, sizeof(DataTy)*Ni, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x1, x1, sizeof(DataTy)*Ni, cudaMemcpyHostToDevice);

  // Compute grid and block size.
  int tileSize = 128;
  int nTiles = Ni/tileSize + (Ni%tileSize == 0?0:1);
  printf("tileSize = %d, nTiles = %d\n", tileSize, nTiles);
  //int gridHeight = Nj/tileSize + (Nj%tileSize == 0?0:1);
  //int gridWidth = Ni/tileSize + (Ni%tileSize == 0?0:1);
  //printf("w=%d, h=%d\n",gridWidth,gridHeight);
  //dim3 dGrid(gridHeight, gridWidth);
  //dim3 dBlock(tileSize, tileSize);

  DataTy *d_x_old = d_x0;
  DataTy *d_x_new = d_x1;
  for (int k = 0; k <= iter; k ++) {
    jacobi_kernel<<< nTiles, tileSize >>>(d_A, d_b, d_x_old, d_x_new, Ni, Nj);
    cudaMemcpy(x1, d_x_new, sizeof(DataTy)*Ni, cudaMemcpyDeviceToHost);
    // swap pointers
    DataTy *d_x = d_x_new;
    d_x_new = d_x_old;
    d_x_old = d_x;
    double r_norm = 0.0;
    // TODO: move this onto GPU
    // compute the norm of A*x-b.
    #pragma omp parallel for reduction(+:r_norm)
    for (int i = 0; i < n; i++ ) {
      double r = - b[i];
      for (int j = 0; j < n; j++ )
        r += A[i+j*n] * x1[j];
      r_norm += r * r;
    }
    r_norm = sqrt ( r_norm );
    if (( k <= 20 ) | ( ( k % 20 ) == 0 ))
      printf ("  %3d  %g\n", k, r_norm);
  }
  // Data <- device
  if (iter%2) cudaMemcpy(x0, d_x1, sizeof(DataTy)*Ni, cudaMemcpyDeviceToHost);
  else cudaMemcpy(x0, d_x0, sizeof(DataTy)*Ni, cudaMemcpyDeviceToHost);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "jacobi_gpu_base");

  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_x0);
  cudaFree(d_x1);
}

