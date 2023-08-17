// Copyright 2020 MIT
// Author: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

void MISSolver(Graph &g, VertexList &mis) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk MIS (" << num_threads << " threads)\n";

  Timer t;
  t.Start();
  // add your code here
  t.Stop();
  std::cout << "runtime [cilk_base] = " << t.Seconds() << " sec\n";
}

