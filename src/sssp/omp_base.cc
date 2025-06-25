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
  auto ne = g.E();
  std::vector<vidType> frontier(ne);
  std::vector<vidType> next_frontier(ne);
  frontier[0] = source;
  size_t frontier_size = 1;
  while (frontier_size > 0) {
    ++ iter;
    std::cout << "iteration=" << iter << ", frontier size: " << frontier_size << "\n";
    size_t index = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < frontier_size; i++) {
      auto u = frontier[i];
      auto begin = g.edge_begin(u);
      auto end = g.edge_end(u);
      for (auto e = begin; e < end; ++e) {
        auto v = g.get_edge_dst(e);
        auto wt = edge_weights[e];
        auto new_dist = distances[u] + wt;
        auto old_dist = distances[v];
        if (new_dist < old_dist) {
          if (atomicMin(distances[v], new_dist)) {
            size_t pos = fetch_and_add(index, 1);
            next_frontier[pos] = v;
          }
        }
      }
    }
    frontier_size = index;
    std::swap(frontier, next_frontier);
  }
  //std::cout << "iterations = " << iter << "\n";
}
