#include <omp.h>
#include <vector>
#include "BaseGraph.hh"
#include "platform_atomics.h"

void BFSSolver(BaseGraph &g, vidType source, int* depths) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP BFS (" << num_threads << " threads)\n";

  int iter = 0;
  depths[source] = 0;
  std::vector<vidType> frontier(g.V());
  std::vector<vidType> next_frontier(g.V());
  frontier[0] = source;
  size_t frontier_size = 1;
  
  while (frontier_size > 0) {
    ++ iter;
    std::cout << "iteration=" << iter << ", frontier_size=" << frontier_size << "\n";
    size_t index = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < frontier_size; i++) {
      auto u = frontier[i];
      //printf("v= %d, deg = %d\n", u, g.get_degree(u));
      for (auto v : g.N(u)) {
        if (depths[v] == -1) {
          // make sure each unique neighbor is inserted to the frontier only once
          if (compare_and_swap(depths[v], -1, depths[u] + 1)) {
            // avoid data race on frontier insertion
            size_t pos = fetch_and_add(index, 1);
            next_frontier[pos] = v;
          }
        }
      }
    }
    frontier_size = index;
    std::swap(frontier, next_frontier);
  }
  std::cout << "iterations = " << iter << "\n";
}

