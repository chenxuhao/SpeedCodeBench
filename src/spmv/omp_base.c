#include <omp.h>
#include <stdint.h>
#include <stddef.h>

typedef float T;
typedef uint32_t vidType;
typedef int64_t eidType;

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
  for (vidType i = 0; i < m; i++){
    eidType row_begin = Ap[i];   // 8 bytes
    eidType row_end   = Ap[i+1]; // 8 bytes
    T sum = y[i];          // 4 bytes
    for (eidType jj = row_begin; jj < row_end; jj++) {
      vidType j = Aj[jj];        //column index 4 bytes
      sum += x[j] * Ax[jj];   // 4 + 4 = 8 bytes
    }
    y[i] = sum; 
  }
}

