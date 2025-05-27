#include "BaseGraph.hh"

typedef float T;

void SpmvSolver(size_t m, size_t nnz, const eidType *Ap, const vidType *Aj, const T *Ax, const T *x, T *y) {
  printf("Serial SpMV solver\n");
  for (vidType i = 0; i < m; i++){
    auto row_begin = Ap[i];
    auto row_end   = Ap[i+1];
    auto sum = y[i];
    for (auto jj = row_begin; jj < row_end; jj++) {
      auto j = Aj[jj];  //column index
      sum += x[j] * Ax[jj];
    }
    y[i] = sum; 
  }
}

