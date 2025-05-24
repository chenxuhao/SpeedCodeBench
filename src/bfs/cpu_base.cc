// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "BaseGraph.hh"

void BFSSolver(BaseGraph &g, vidType source, int* depth) {
  std::cout << "Queue (FIFO) based serial BFS\n";
  std::vector<vidType> to_visit;
  depth[source] = 0;
  to_visit.reserve(g.V());
  to_visit.push_back(source);
  for (std::vector<vidType>::iterator it = to_visit.begin(); it != to_visit.end(); it++) {
    auto src = *it;
    //for (auto dst : g.N(src)) {
    auto adj = g.get_adj(src);
    auto deg = g.get_degree(src);
    for (vidType i = 0; i < deg; i++) {
      auto dst = adj[i];
      if (depth[dst] == -1) {
        depth[dst] = depth[src] + 1;
        to_visit.push_back(dst);
      }
    }
  }
}

