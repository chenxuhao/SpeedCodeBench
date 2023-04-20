#include <omp.h>
#include <stdio.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#define TILE_SIZE 4
#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

extern "C"
void stencil(float c0, float c1, float *A0, float *Anext,
             const int nx, const int ny, const int nz, const int niter) {
  int i;
  int j, k;
  int ex, ey, ez;//element index
  int x_grid, y_grid, z_grid;
  int x_bound, y_bound, z_bound;
  x_grid = (nx-2+TILE_SIZE-1)/TILE_SIZE;
  x_bound = (nx-2)%TILE_SIZE;
  y_grid = (ny-2+TILE_SIZE-1)/TILE_SIZE;
  y_bound = (ny-2)%TILE_SIZE;
  z_grid = (nz-2+TILE_SIZE-1)/TILE_SIZE;
  z_bound = (nz-2)%TILE_SIZE;

  double start = omp_get_wtime();
  int t;
  //int tx, ty, tz;//tile index
  for (t = 0; t < niter; t++) {
    cilk_for (int tz=0; tz<z_grid; tz++) {
      cilk_for (int ty=0; ty<y_grid; ty++) {
        cilk_for (int tx=0; tx<x_grid; tx++) {
          //check bound
          int xb = TILE_SIZE; //bound of ex
          int yb = TILE_SIZE; //bound of ey
          int zb = TILE_SIZE; //bound of ez
          if((tx==(x_grid-1)) && (x_bound!=0))
            xb = x_bound;
          if((ty==(y_grid-1)) && (y_bound!=0))
            yb = y_bound;
          if((tz==(z_grid-1)) && (z_bound!=0))
            zb = z_bound;
          for(ex=0;ex<xb;ex++) {
            for(ey=0;ey<yb;ey++) {
              for(ez=0;ez<zb;ez++) {
                i = tx*TILE_SIZE+ex+1;
                j = ty*TILE_SIZE+ey+1;
                k = tz*TILE_SIZE+ez+1;
                Anext[Index3D (nx, ny, i, j, k)] = 
                  (A0[Index3D (nx, ny, i, j, k + 1)] +
                   A0[Index3D (nx, ny, i, j, k - 1)] +
                   A0[Index3D (nx, ny, i, j + 1, k)] +
                   A0[Index3D (nx, ny, i, j - 1, k)] +
                   A0[Index3D (nx, ny, i + 1, j, k)] +
                   A0[Index3D (nx, ny, i - 1, j, k)])*c1
                  - A0[Index3D (nx, ny, i, j, k)]*c0;
              }//end for ez
            }// end for ey
          }//end for ex
        }
      }
    }
    float *temp = A0;
    A0 = Anext;
    Anext = temp;
  }
  float *temp = A0;
  A0 = Anext;
  Anext = temp;
  double end = omp_get_wtime();
  printf("runtime = %f sec\n", end - start);
}

