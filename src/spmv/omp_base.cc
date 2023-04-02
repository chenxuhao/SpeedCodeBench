// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "timer.h"
//#include "spmv_util.h"

template <typename T>
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
  for (int i = 0; i < m; i++){
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
  //float gbyte = bytes_per_spmv(m, nnz);
  //float GFLOPs = (time == 0) ? 0 : (2 * nnz / time);
  //float GBYTEs = (time == 0) ? 0 : (gbyte / time);
  //printf("\truntime [omp_base] = %.4f s ( %5.2f GFLOP/s %5.1f GB/s)\n", time, GFLOPs, GBYTEs);
  std::cout << "runtime [base] = " << t.Seconds() << " sec\n";
  return;
}

typedef float ValueT;
int main(int argc, char *argv[]) {
  printf("Sparse Matrix-Vector Multiplication\n");
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <graph-prefix>\n";
    std::cout << "Example: " << argv[0] << " inputs/citeseer\n";
    exit(1);
  }
  Graph g(argv[1]);
  std::vector<ValueT> x(g.V(), 0);
  std::vector<ValueT> y(g.V(), 0);
  //srand(13);
  for(int i = 0; i < g.V(); i++) {
    //x[i] = rand() / (RAND_MAX + 1.0);
    //y[i] = rand() / (RAND_MAX + 1.0);
    x[i] = 0.3;
  }

  SpmvSolver<ValueT>(g, x.data(), y.data());
  //SpmvVerifier(g, x, y_ref, y);
  return 0;
}

