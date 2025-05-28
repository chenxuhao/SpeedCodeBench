#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>
#include <stdint.h>
#include <stddef.h>

typedef float T;
typedef uint32_t vidType;
typedef int64_t eidType;

extern "C"
void SpmvSolver(size_t m, size_t nnz, const eidType *Ap, const vidType *Aj, const T *Ax, const T *x, T *y) {
  //int num_threads = __cilkrts_get_nworkers();
  //std::cout << "Cilk SpMV (" << num_threads << " threads)\n";
 
  #pragma cilk grainsize 64
  cilk_for (vidType i = 0; i < m; i++) {
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

