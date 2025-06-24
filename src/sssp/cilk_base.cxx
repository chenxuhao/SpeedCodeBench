#include <climits>
#include "BaseGraph.hh"
#include "sliding_queue.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

typedef int T;
#define kDistInf UINT_MAX/2

void SSSPSolver(BaseGraph &g, T* weights, vidType source, T *distances) {
  auto nthreads = __cilkrts_get_nworkers();
  std::cout << "Cilk SSSP (" << nthreads << " threads)\n";
  int iter = 0;
  distances[source] = 0;
  //SlidingQueue<vidType> queue(g.V());
  SlidingQueue<vidType> queue(g.E());
  queue.push_back(source);
  queue.slide_window();
  while (!queue.empty()) {
    ++ iter;
    LocalBuffer<vidType> lqueues(queue, nthreads);
    vidType* ptr = queue.begin();
    //std::cout << "iteration=" << iter << ", frontier_size=" << queue.size() << "\n";
    #pragma cilk grainsize 64
    cilk_for (int i = 0; i < queue.size(); i++) {
      auto tid = __cilkrts_get_worker_number();
      auto v = ptr[i];
      auto offset = g.edge_begin(v);
      auto dist_v = distances[v];
      for (auto u : g.N(v)) {
        auto wt = weights[offset++];
        auto new_dist = dist_v + wt;
        auto old_dist = distances[u];
        if (new_dist < old_dist) {
          if (atomicMin(distances[u], new_dist)) {
            lqueues.push_back(tid, u);
          }
        }
      }
    }
    lqueues.collect();
    queue.slide_window();
  }
  std::cout << "iterations = " << iter << "\n";
}
