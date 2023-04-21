// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void BFS(Graph &g, vidType source, vidType* depth) {
  std::vector<vidType> to_visit;
  depth[source] = 0;
  to_visit.reserve(g.V());
  to_visit.push_back(source);
  for (std::vector<vidType>::iterator it = to_visit.begin(); it != to_visit.end(); it++) {
    auto src = *it;
    for (auto dst : g.N(src)) {
      if (depth[dst] == MYINFINITY) {
        depth[dst] = depth[src] + 1;
        to_visit.push_back(dst);
      }
    }
  }
}

void BFSVerifier(Graph &g, vidType source, vidType *depth_to_test) {
  std::cout << "Verifying BFS...\n";
  std::vector<vidType> depth(g.V(), MYINFINITY);
  BFS(g, source, depth.data());
  // Report any mismatches
  bool all_ok = true;
  for (vidType n = 0; n < g.V(); n ++) {
    if (depth_to_test[n] != depth[n]) {
      //std::cout << n << ": " << depth_to_test[n] << " != " << depth[n] << std::endl;
      all_ok = false;
    }
  }
  if (all_ok) std::cout << "Correct\n";
  else std::cout << "Wrong\n";
}

