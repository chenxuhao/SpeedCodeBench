#include <omp.h>
#include <stdio.h>

#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

__global__ void stencil_kernel(float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz) {
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if (i > 0) {
    Anext[Index3D (nx, ny, i, j, k)] = 
      (A0[Index3D (nx, ny, i, j, k + 1)] +
       A0[Index3D (nx, ny, i, j, k - 1)] +
       A0[Index3D (nx, ny, i, j + 1, k)] +
       A0[Index3D (nx, ny, i, j - 1, k)] +
       A0[Index3D (nx, ny, i + 1, j, k)] +
       A0[Index3D (nx, ny, i - 1, j, k)])*c1
      - A0[Index3D (nx, ny, i, j, k)]*c0;
  }
}

extern "C"
void stencil(float c0, float c1, float *A0, float *Anext,
             const int nx, const int ny, const int nz, const int niter) {
  size_t size = nx * ny * nz;
  float *d_A0, *d_Anext;
  cudaMalloc((void **)&d_A0, size * sizeof(float));
  cudaMalloc((void **)&d_Anext, size * sizeof(float));
  cudaMemcpy(d_A0, A0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Anext, d_A0, size * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaThreadSynchronize();
  dim3 block(nx-1, 1, 1);
  dim3 grid(ny-2, nz-2,1);

  double start = omp_get_wtime();
  int t;
  for (t = 0; t < niter; t++) {
    stencil_kernel<<<grid, block>>>(c0, c1, d_A0, d_Anext, nx, ny, nz);
    float *d_temp = d_A0;
    d_A0 = d_Anext;
    d_Anext = d_temp;
  }
  float *d_temp = d_A0;
  d_A0 = d_Anext;
  d_Anext = d_temp;
  cudaMemcpy(Anext, d_Anext, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  double end = omp_get_wtime();

  printf("runtime = %f sec\n", end - start);
  cudaFree(d_A0);
  cudaFree(d_Anext);
}
