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
  float gbyte = bytes_per_spmv(m, nnz) / 10e9;
  assert(time > 0.);
  float GFLOPs = 2*nnz / time / 10e9;
  float GBYTEs = gbyte / time;
  std::cout << "runtime [serial] = " << t.Seconds() << " sec\n";
  printf("Throughput: compute %5.2f GFLOP/s, memory %5.1f GB/s\n", GFLOPs, GBYTEs);
  return;
}

