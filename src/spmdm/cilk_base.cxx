// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "ctimer.h"
#include <stdint.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

typedef float T;
typedef uint32_t vidType;
typedef int64_t eidType;

// A: m x m 
// B: m x n
// BT:n x m
// C: m x n
extern "C"
void SpmDm(char transa, char transb, 
           vidType m, eidType nnz, int n,
           T alpha, const eidType *Ap,
           const vidType *Aj, const T *Ax, 
           int lda, const T *BT, int ldb, 
           T beta, T *C, int ldc) {
  int num_threads = __cilkrts_get_nworkers();
  printf("Cilk SpMDM (%d threads)\n", num_threads);
  ctimer_t t;
  ctimer_start(&t);
  cilk_for (vidType i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      T sum = 0;
      for (auto off = Ap[i]; off < Ap[i+1]; off++) {
        auto k = Aj[off];
        auto value = Ax[off]; // A[i][k]
        sum += value * BT[j*m + k]; // A[i][k] * BT[j][k]
      }
      C[i*n + j] = sum; // C[i][j]
    }
  }
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "SpMDM-cilk_base");
}

