// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "timer.h"
#include "spmv_util.h"
typedef float T;

void SpmvSolver(GraphF &g, const T *x, T *y) {
  auto m = g.V();
  auto nnz = g.E();
  auto Ap = g.in_rowptr();
  auto Aj = g.in_colidx();
  auto Ax = g.get_elabel_ptr();
  printf("Serial SpMV solver\n");
  Timer t;
  t.Start();
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
  t.Stop();
  double time = t.Seconds();
  std::cout << "runtime [serial] = " << t.Seconds() << " sec\n";
  print_throughput(m, nnz, time);
}

