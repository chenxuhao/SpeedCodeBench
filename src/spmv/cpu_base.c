#include <stdint.h>
#include <stddef.h>

typedef float T;
typedef uint32_t vidType;
typedef int64_t eidType;

void SpmvSolver(size_t m, size_t nnz, const eidType *Ap, const vidType *Aj, const T *Ax, const T *x, T *y) {
  for (vidType i = 0; i < m; i++){
    eidType row_begin = Ap[i];
    eidType row_end   = Ap[i+1];
    T sum = y[i];
    for (eidType jj = row_begin; jj < row_end; jj++) {
      vidType j = Aj[jj];  //column index
      sum += x[j] * Ax[jj];
    }
    y[i] = sum; 
  }
}

