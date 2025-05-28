#include <stdio.h>
#include <string.h>
#include <ctimer.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <vector>

typedef float T;

#define TILE_SIZE 48

//m is matArow (hA), n is matBcol (wB), and k is matAcol (wA)
extern "C"
void sgemm(char transa, char transb,
           int m, int n, int k, 
           float alpha, const float *_A, int lda,
           const float *_B, int ldb, float beta,
           float *_C, int ldc) {
  int num_threads = __cilkrts_get_nworkers();
  printf("Cilk SGEMM (%d threads)\n", num_threads);
 
  if ((transa != 'N') && (transa != 'n')) {
    printf("unsupported value of 'transa' in regtileSgemm()\n");
    return;
  }
  if ((transb != 'T') && (transb != 't')) {
    printf("unsupported value of 'transb' in regtileSgemm()\n");
    return;
  }
  int tx, ty; // element index
  float Asub, Bsub, Csub;
  int hA_grid, wB_grid, wA_grid;
  int hA_bound, wB_bound, wA_bound;
  int a, b;
  //height and width of A, B, C
  int hA,wA, wB, hC, wC;
  hA = m;
  wA = k;
  wB = n;
  hC = m;
  wC = n;

  //clear C
  memset(_C, 0, hC*wC*sizeof(float));

  std::vector<T>A(_A, _A+m*k);
  std::vector<T>B(_B, _B+k*n);
  std::vector<T>C(_C, _C+m*n);

  ctimer_t t;
  ctimer_start(&t);
  hA_grid = (hA+TILE_SIZE-1)/TILE_SIZE;
  hA_bound = hA%TILE_SIZE;
  wB_grid = (wB+TILE_SIZE-1)/TILE_SIZE;
  wB_bound = wB%TILE_SIZE;
  wA_grid = (wA+TILE_SIZE-1)/TILE_SIZE;
  wA_bound = wA%TILE_SIZE;

  // bx, by: tile index
  //for each block in the whole matrix C
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int by=0; by<hA_grid; by++) {
    for (int bx=0; bx<wB_grid; bx++) {
      //for each block in the same row of martix A (or the same column of matrix B)
      for (a=0; a<wA_grid; a++) {
        //check bound
        int yb = TILE_SIZE; //bound of ty
        int xb = TILE_SIZE; //bound of tx
        int bb = TILE_SIZE; //bound of b
        if ((by==(hA_grid-1)) && (hA_bound!=0))
          yb = hA_bound;
        if ((bx==(wB_grid-1)) && (wB_bound!=0))
          xb = wB_bound;
        if ((a==(wA_grid-1)) && (wA_bound!=0))
          bb = wA_bound;

        //for each elements in the block
        for (ty=0; ty<yb; ty++) {
          for (tx=0; tx<xb; tx++) {
            Csub= 0.0f;
            int idy = by*TILE_SIZE+ty;
            int idx = bx*TILE_SIZE+tx;
            int blockNum = a*TILE_SIZE;
            for (b=0; b<bb; ++b) {
              Asub = A[idy+hA*(blockNum+b)];//(y, x) = (idy, (blockNum+b))
              Bsub = B[(blockNum+b)*wB+idx];//(y, x) = ((blockNum+b), idx)
              Csub += Asub * Bsub;
            }//end for b
            C[idy+hC*idx] += Csub;//(y, x) = (idy, idx)
          }//end for tx
        }//end for ty
      }//end for a
    }//end for bx
  }//end for by
 
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "SGEMM-zera_base-kernel");

  std::copy(C.begin(), C.end(), _C);
}

