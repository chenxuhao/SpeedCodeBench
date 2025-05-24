// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "../common/BaseGraph.cc"
#include "sliding_queue.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

void BFSSolver(BaseGraph &g, vidType source, int *depths) {
  int iter = 0;
  auto nthreads = __cilkrts_get_nworkers();
  std::cout << "Cilk BFS (" << nthreads << " threads)\n";
  SlidingQueue<vidType> queue(g.V());
  queue.push_back(source);
  queue.slide_window();
  depths[source] = 0;

  auto nv = g.V();
  const eidType* _verts = g.rowptr(); // get row pointers array
  const vidType* _edges = g.colidx(); // get column indices array
  std::vector<eidType>verts(_verts, _verts+nv+1);
  std::vector<vidType>edges(_edges, _edges+g.E());
 
  while (!queue.empty()) {
    ++ iter;
    LocalBuffer<vidType> lqueues(queue, nthreads);
    vidType* ptr = queue.begin();
    std::cout << "iteration=" << iter << ", frontier_size=" << queue.size() << "\n";
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (int i = 0; i < queue.size(); i++) {
      auto tid = __cilkrts_get_worker_number();
      auto u = ptr[i];
      auto u_adj = &edges[verts[u]];
      auto u_deg = vidType(verts[u+1] - verts[u]);
      for (vidType j = 0; j < u_deg; j++) {
        auto v = u_adj[i];
      //for (auto v : g.N(u)) {
        int curr_val = depths[v];
        if (curr_val == -1) {
          if (compare_and_swap(depths[v], -1, depths[u] + 1)) {
            lqueues.push_back(tid, v);
          }
        }
      }
    }
    lqueues.collect();
    queue.slide_window();
  }
  std::cout << "iterations = " << iter << "\n";
}

