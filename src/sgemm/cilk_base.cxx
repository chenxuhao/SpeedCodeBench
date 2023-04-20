#include <iostream>
#include <string.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#define BLOCK_SIZE 48

//m is matArow (hA), n is matBcol (wB), and k is matAcol (wA)
void sgemm(char transa, char transb,
           int m, int n, int k, 
           float alpha, const float *A, int lda,
           const float *B, int ldb, float beta,
           float *C, int ldc ) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk SGEMM (" << num_threads << " threads)\n";
 
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()\n";
    return;
  }
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()\n";
    return;
  }
  int tx, ty;// element index
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
  memset(C, 0, hC*wC*sizeof(float));
  hA_grid = (hA+BLOCK_SIZE-1)/BLOCK_SIZE;
  hA_bound = hA%BLOCK_SIZE;
  wB_grid = (wB+BLOCK_SIZE-1)/BLOCK_SIZE;
  wB_bound = wB%BLOCK_SIZE;
  wA_grid = (wA+BLOCK_SIZE-1)/BLOCK_SIZE;
  wA_bound = wA%BLOCK_SIZE;

  // bx, by: tile index
  //for each block in the whole matrix C
  cilk_for (int by=0; by<hA_grid; by++) {
    cilk_for (int bx=0; bx<wB_grid; bx++) {
      //for each block in the same row of martix A (or the same column of matrix B)
      for (a=0; a<wA_grid; a++) {
        //check bound
        int yb = BLOCK_SIZE; //bound of ty
        int xb = BLOCK_SIZE; //bound of tx
        int bb = BLOCK_SIZE; //bound of b
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
            int idy = by*BLOCK_SIZE+ty;
            int idx = bx*BLOCK_SIZE+tx;
            int blockNum = a*BLOCK_SIZE;
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
}

