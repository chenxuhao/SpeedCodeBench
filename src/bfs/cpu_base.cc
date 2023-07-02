// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void BFSSolver(Graph &g, vidType source, int* depth) {
  std::cout << "Queue (FIFO) based serial BFS\n";
  std::vector<vidType> to_visit;
  Timer t;
  t.Start();
  depth[source] = 0;
  to_visit.reserve(g.V());
  to_visit.push_back(source);
  for (std::vector<vidType>::iterator it = to_visit.begin(); it != to_visit.end(); it++) {
    auto src = *it;
    for (auto dst : g.N(src)) {
      if (depth[dst] == -1) {
        depth[dst] = depth[src] + 1;
        to_visit.push_back(dst);
      }
    }
  }
  t.Stop();
  std::cout << "runtime [serial] = " << t.Seconds() << " sec\n";
}

