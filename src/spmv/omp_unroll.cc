// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "timer.h"
#include "spmv_util.h"
typedef float T;
#define UNROLL 3

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
  t.Stop();

  double time = t.Seconds();
  std::cout << "runtime [omp_unroll] = " << t.Seconds() << " sec\n";
  print_throughput(m, nnz, time);
}

