// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include <omp.h>
#include <math.h>
#include "pr.h"
#include "BaseGraph.hh"
#include "platform_atomics.h"

void PRSolver(BaseGraph &g, score_t *scores) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP PangeRank (" << num_threads << " threads)\n";
  auto nv = g.V();
  const score_t base_score = (1.0f - kDamp) / nv;
  std::vector<score_t> sums(nv, 0);
  int iter;
  for (iter = 0; iter < MAX_ITER; iter ++) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (vidType src = 0; src < nv; src ++) {
      score_t contribution = scores[src] / (score_t)g.get_degree(src);
      for (auto dst : g.N(src)) {
        #pragma omp atomic
        sums[dst] = sums[dst] + contribution;
        //fetch_and_add(sums[dst], contribution);
        //sums[dst].fetch_add(contribution, std::memory_order_relaxed);
      }
    }
    double error = 0;
    #pragma omp parallel for reduction(+ : error)
    for (vidType u = 0; u < nv; u ++) {
      score_t new_score = base_score + kDamp * sums[u];
      error += fabs(new_score - scores[u]);
      scores[u] = new_score;
      sums[u] = 0;
    }
    printf(" %2d    %lf\n", iter+1, error);
    if (error < EPSILON) break;
  }
  std::cout << "iterations = " << iter+1 << ".\n";
}
