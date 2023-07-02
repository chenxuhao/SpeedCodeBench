// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "sliding_queue.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

void BFSSolver(Graph &g, vidType source, int *depths) {
  int iter = 0;
  auto nthreads = __cilkrts_get_nworkers();
  std::cout << "Cilk BFS (" << nthreads << " threads)\n";
  SlidingQueue<vidType> queue(g.V());
  Timer t;
  t.Start();
  queue.push_back(source);
  queue.slide_window();
  depths[source] = 0;
  while (!queue.empty()) {
    ++ iter;
    LocalBuffer<vidType> lqueues(queue, nthreads);
    vidType* ptr = queue.begin();
    std::cout << "iteration=" << iter << ", frontier_size=" << queue.size() << "\n";
    cilk_for (int i = 0; i < queue.size(); i++) {
      auto tid = __cilkrts_get_worker_number();
      auto u = ptr[i];
      for (auto v : g.N(u)) {
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
  t.Stop();
  std::cout << "iterations = " << iter << "\n";
  std::cout << "runtime [cilk_base] = " << t.Seconds() << " sec\n";
}

