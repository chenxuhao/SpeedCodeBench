// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "timer.h"
#include "spmv_util.h"
#include <omp.h>

typedef float T;

void SpmvSolver(GraphF &g, const T *x, T *y) {
  auto m = g.V();
  auto nnz = g.E();
  auto Ap = g.in_rowptr();
  auto Aj = g.in_colidx();
  auto Ax = g.get_elabel_ptr();
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP SpMV solver (%d threads) ...\n", num_threads);

  Timer t;
  t.Start();
  //#pragma omp parallel for
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
  t.Stop();

  double time = t.Seconds();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  print_throughput(m, nnz, time);
}

