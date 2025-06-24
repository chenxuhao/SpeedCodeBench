#include <omp.h>
#include <climits>
#include "BaseGraph.hh"
#include "sliding_queue.h"

typedef int T;
#define kDistInf UINT_MAX/2
void SSSPSolver(BaseGraph &g, T *weights, vidType source, T *distances) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP SSSP solver (%d threads)\n", num_threads);
  distances[source] = 0;
  int iter = 0;
  SlidingQueue<vidType> queue(g.E());
  queue.push_back(source);
  queue.slide_window();
  while (!queue.empty()) {
    ++ iter;
    //std::cout << "iteration=" << iter << ", frontier_size=" << queue.size() << "\n";
    LocalBuffer<vidType> lqueues(queue, num_threads);
    vidType* ptr = queue.begin();
    #pragma omp parallel for
    for (size_t i = 0; i < queue.size(); i++) {
      auto tid = omp_get_thread_num();
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
}