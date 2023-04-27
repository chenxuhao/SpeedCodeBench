// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

void CCSolver(Graph &g, comp_t *comp) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk Connected Components (" << num_threads << " threads)\n";
  cilk_for (vidType n = 0; n < g.V(); n ++) comp[n] = n;
  bool change = true;
  int iter = 0;

  Timer t;
  t.Start();
  while (change) {
    change = false;
    iter++;
    //printf("Executing iteration %d ...\n", iter);
    cilk_for (vidType src = 0; src < g.V(); src ++) {
      auto comp_src = comp[src];
      for (auto dst : g.N(src)) {
        auto comp_dst = comp[dst];
        if (comp_src == comp_dst) continue;
        // Hooking condition so lower component ID wins independent of direction
        int high_comp = comp_src > comp_dst ? comp_src : comp_dst;
        int low_comp = comp_src + (comp_dst - high_comp);
        if (high_comp == comp[high_comp]) {
          change = true;
          comp[high_comp] = low_comp;
        }
      }
    }
    cilk_for (vidType n = 0; n < g.V(); n++) {
      while (comp[n] != comp[comp[n]]) {
        comp[n] = comp[comp[n]];
      }
    }
  }
  t.Stop();
  std::cout << "iterations = " << iter << "\n";
  std::cout << "runtime [cilk_base] = " << t.Seconds() << " seconds\n";
  return;
}
