// Copyright 2020 MIT
// Author: Xuhao Chen <cxh@mit.edu>
#include <omp.h>
#include "graph.h"

void MISSolver(Graph &g, VertexList &mis) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP MIS (" << num_threads << " threads) ...\n";

  Timer t;
  t.Start();
  // add your code here
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
}

