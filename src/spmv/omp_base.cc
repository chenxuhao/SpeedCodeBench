#include <omp.h>
#include "BaseGraph.hh"

typedef float T;

void SpmvSolver(size_t m, size_t nnz, const eidType *Ap, const vidType *Aj, const T *Ax, const T *x, T *y) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP SpMV solver (%d threads) ...\n", num_threads);

  #pragma omp parallel for schedule (dynamic, 1024)
  for (vidType i = 0; i < m; i++){
    auto row_begin = Ap[i];   // 8 bytes
    auto row_end   = Ap[i+1]; // 8 bytes
    auto sum = y[i];          // 4 bytes
    for (auto jj = row_begin; jj < row_end; jj++) {
      auto j = Aj[jj];        //column index 4 bytes
      sum += x[j] * Ax[jj];   // 4 + 4 = 8 bytes
    }
    y[i] = sum; 
  }
}

