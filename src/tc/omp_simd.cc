#include <omp.h>
#include "BaseGraph.hh"
#include "intersect.h"

void TCSolver(Graph &g, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP SIMD TC (" << num_threads << " threads)\n";
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.get_adj(u);
    auto deg_u = g.get_degree(u);
    for (vidType i = 0; i < deg_u; i++) {
      auto v = adj_u[i];
      auto adj_v = g.get_adj(v);
      auto deg_v = g.get_degree(v);
      //counter += (uint64_t)set_intersect(deg_u, deg_v, adj_u, adj_v);
      counter += (uint64_t)SetIntersection::get_num(adj_u, deg_u, adj_v, deg_v);
    }
  }
  total = counter;
  return;
}

