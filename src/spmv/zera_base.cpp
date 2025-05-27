#include "BaseGraph.hh"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "../common/BaseGraph.cc"

typedef float T;

void SpmvSolver(size_t m, size_t nnz, const eidType *_Ap, const vidType *_Aj, const T *_Ax, const T *_x, T *_y) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk SpMV (" << num_threads << " threads)\n";

  std::vector<eidType>Ap(_Ap, _Ap+m+1);
  std::vector<vidType>Aj(_Aj, _Aj+nnz);
  std::vector<T>Ax(_Ax, _Ax+nnz);
  std::vector<T>x(_x, _x+m);
  std::vector<T>y(_y, _y+m);
 
  [[tapir::target("cuda"), tapir::grain_size(1)]]
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
  std::copy(y.begin(), y.end(), _y);
}

