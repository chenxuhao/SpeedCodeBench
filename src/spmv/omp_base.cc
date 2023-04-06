// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "timer.h"
#include "spmv_util.h"
typedef int32_t T;

void SpmvSolver(Graph &g, const T *x, T *y) {
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
  #pragma omp parallel for
  for (vidType i = 0; i < m; i++){
    auto row_begin = Ap[i];
    auto row_end   = Ap[i+1];
    auto sum = y[i];
    //#pragma omp simd reduction(+:sum)
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
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  printf("Throughput: compute %5.2f GFLOP/s, memory %5.1f GB/s\n", GFLOPs, GBYTEs);
  return;
}

