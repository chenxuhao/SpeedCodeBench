#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))
void stencil(float c0,float c1, float *A0,float * Anext,
             const int nx, const int ny, const int nz) {
  int i, j, k;
  for (i=1;i<nx-1;i++) {
    for (j=1;j<ny-1;j++) {
      for (k=1;k<nz-1;k++) {
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
  }
}

