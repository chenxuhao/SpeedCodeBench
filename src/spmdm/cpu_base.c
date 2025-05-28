#include <stdint.h>

typedef float T;
typedef uint32_t vidType;
typedef int64_t eidType;

// A: m x m 
// B: m x n
// BT:n x m
// C: m x n
void SpmDm(char transa, char transb, 
           vidType m, eidType nnz, int n,
           T alpha, const eidType *Ap,
           const vidType *Aj, const T *Ax, 
           int lda, const T *BT, int ldb, 
           T beta, T *C, int ldc) {
  for (vidType i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      T sum = 0;
      for (eidType off = Ap[i]; off < Ap[i+1]; off++) {
        vidType k = Aj[off];
        //if (k >= m) printf("k=%u, off=%ld, i=%u, j=%d\n", k, off, i, j);
        T value = Ax[off]; // A[i][k]
        sum += value * BT[j*m + k]; // A[i][k] * BT[j][k]
      }
      C[i*n + j] = sum; // C[i][j]
    }
  }
}

