// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "graph.h"
typedef float T;

void gauss_seidel(eidType *Ap, vidType *Aj, vidType *indices, T *Ax, T *x, T *b, 
                  int row_start, int row_stop, int row_step) {
  #pragma omp parallel for
  for (int i = row_start; i < row_stop; i += row_step) {
    auto inew = indices[i];
    auto row_begin = Ap[inew];
    auto row_end = Ap[inew+1];
    T rsum = 0;
    T diag = 0;
    for (auto jj = row_begin; jj < row_end; jj++) {
      auto j = Aj[jj];  //column index
      if (inew == j) diag = Ax[jj];
      else rsum += x[j] * Ax[jj];
    }
    if (diag != 0) x[inew] = (b[inew] - rsum) / diag;
  }
}

void SymGSSolver(GraphF &g, vidType *indices, T *x, T *b, std::vector<int> color_offsets) {
  auto Ap = g.in_rowptr();
  auto Aj = g.in_colidx();
  auto Ax = g.get_elabel_ptr();
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("Launching OpenMP SymGS solver (%d threads) ...\n", num_threads);
  Timer t;
  t.Start();
  for(size_t i = 0; i < color_offsets.size()-1; i++)
    gauss_seidel(Ap, Aj, indices, Ax, x, b, color_offsets[i], color_offsets[i+1], 1);
  for(size_t i = color_offsets.size()-1; i > 0; i--)
    gauss_seidel(Ap, Aj, indices, Ax, x, b, color_offsets[i-1], color_offsets[i], 1);
  t.Stop();
  printf("runtime [symgs_omp_base] = %f ms.\n", t.Seconds());
  return;
}

