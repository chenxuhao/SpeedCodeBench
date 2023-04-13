#include <stdio.h>

void sgemm(char transa, char transb, 
           int m, int n, int k, 
           float alpha, const float *A, 
           int lda, const float *B, int ldb, 
           float beta, float *C, int ldc) {
  if ((transa != 'N') && (transa != 'n')) {
    printf("unsupported value of 'transa' in regtileSgemm()\n");
    return;
  }
  if ((transb != 'T') && (transb != 't')) {
    printf("unsupported value of 'transb' in regtileSgemm()\n");
    return;
  }
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}
