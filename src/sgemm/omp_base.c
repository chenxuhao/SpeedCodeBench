#include <stdio.h>
#include <string.h>
#define TILE_SIZE 48

//m is matArow (hA), n is matBcol (wB), and k is matAcol (wA)
void sgemm(char transa, char transb,
           int m, int n, int k, 
           float alpha, const float *A, int lda,
           const float *B, int ldb, float beta,
           float *C, int ldc ) {
  if ((transa != 'N') && (transa != 'n')) {
    printf("unsupported value of 'transa' in regtileSgemm()\n");
    return;
  }
  if ((transb != 'T') && (transb != 't')) {
    printf("unsupported value of 'transb' in regtileSgemm()\n");
    return;
  }
  int bx, by;// tile index
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
  hA_grid = (hA+TILE_SIZE-1)/TILE_SIZE;
  hA_bound = hA%TILE_SIZE;
  wB_grid = (wB+TILE_SIZE-1)/TILE_SIZE;
  wB_bound = wB%TILE_SIZE;
  wA_grid = (wA+TILE_SIZE-1)/TILE_SIZE;
  wA_bound = wA%TILE_SIZE;
  //for each block in the whole matrix C
  #pragma omp parallel for shared(A, B, C) private(ty, tx, Asub, Bsub, Csub) collapse(2)//schedule(static)
  for(by=0;by<hA_grid;by++) {
    for(bx=0;bx<wB_grid;bx++) {
      //for each block in the same row of martix A (or the same column of matrix B)
      for(a=0;a<wA_grid;a++) {
	//check bound
	int yb = TILE_SIZE; //bound of ty
	int xb = TILE_SIZE; //bound of tx
	int bb = TILE_SIZE; //bound of b
	if((by==(hA_grid-1)) && (hA_bound!=0))
	  yb = hA_bound;
	if((bx==(wB_grid-1)) && (wB_bound!=0))
	  xb = wB_bound;
	if((a==(wA_grid-1)) && (wA_bound!=0))
	  bb = wA_bound;

	//for each elements in the block
        for(ty=0;ty<yb;ty++) {
          for(tx=0;tx<xb;tx++) {
            Csub= 0.0f;
	    int idy = by*TILE_SIZE+ty;
	    int idx = bx*TILE_SIZE+tx;
		int blockNum = a*TILE_SIZE;
            for(b=0;b<bb;++b) {
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

