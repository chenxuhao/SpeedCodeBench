#include <stdint.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "ctimer.h"
#include <stdint.h>
#include <stddef.h>
#include <vector>

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
           T alpha, const eidType *_Ap,
           const vidType *_Aj, const T *_Ax, 
           int lda, const T *_BT, int ldb, 
           T beta, T *_C, int ldc) {
  int num_threads = __cilkrts_get_nworkers();
  printf("Cilk SpMDM (%d threads)\n", num_threads);

  std::vector<eidType>Ap(_Ap, _Ap+m+1);
  std::vector<vidType>Aj(_Aj, _Aj+nnz);
  std::vector<T>Ax(_Ax, _Ax+nnz);
  std::vector<T>BT(_BT, _BT+m*n);
  std::vector<T>C(_C, _C+m*n);

  ctimer_t t;
  ctimer_start(&t);
  [[tapir::target("cuda"), tapir::grain_size(1)]]
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
  ctimer_print(t, "SpMDM-zera_base-kernel");
  std::copy(C.begin(), C.end(), _C);
}

