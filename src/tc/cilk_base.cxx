// Copyright 2023 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "BaseGraph.hh"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

void TCSolver(BaseGraph &g, uint64_t &total) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk TC (" << num_threads << " threads)\n";
  cilk::opadd_reducer<uint64_t> counter = 0;
  #pragma grainsize = 1
  cilk_for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.get_adj(u);
    auto deg_u = g.get_degree(u);
    for (auto i = 0; i < deg_u; i++) {
      auto v = adj_u[i];
      auto adj_v = g.get_adj(v);
      auto deg_v = g.get_degree(v);
      counter += (uint64_t)set_intersect(deg_u, deg_v, adj_u, adj_v);
    }
  }
  total = counter;
  return;
}

