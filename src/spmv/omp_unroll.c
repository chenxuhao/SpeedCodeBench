#include <omp.h>
#include <stdint.h>

typedef float T;
typedef uint32_t vidType;
typedef int64_t eidType;

typedef float T;
#define UNROLL 3

void SpmvSolver(size_t m, size_t nnz, const eidType *Ap, const vidType *Aj, const T *Ax, const T *x, T *y) {
/*
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP SpMV solver (%d threads) ...\n", num_threads);
*/
  #pragma omp parallel for schedule (dynamic, 1024)
  for (vidType i = 0; i < m; i++) {
    auto row_begin = Ap[i];
    auto row_end   = Ap[i+1];
    auto sum = y[i];
    T uval[UNROLL];
    eidType idx = 0;
    for (int k = 0; k < UNROLL; k++)
      uval[k] = 0.0;
    for (idx = row_begin; idx+UNROLL < row_end; idx+=UNROLL) {
      for (int k = 0; k < UNROLL; k++) {
        auto l = Aj[idx+k];
        uval[k] += x[l] * Ax[idx+k];
      }
    }
    for (; idx < row_end; idx++) {
      auto l = Aj[idx];
      sum += x[l] * Ax[idx];
    }
    for (int k = 0; k < UNROLL; k++)
      sum += uval[k];
    y[i] = sum;
  }
}

