#include <math.h>
#include "pr.h"
#include "BaseGraph.hh"
#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

void PRSolver(BaseGraph &g, score_t *scores) {
  //int num_threads = __cilkrts_get_nworkers();
  //std::cout << "Cilk PangeRank (" << num_threads << " threads)\n";
  auto nv = g.V();
  const score_t base_score = (1.0f - kDamp) / nv;
  score_t *outgoing_contrib = (score_t *) malloc(nv * sizeof(score_t));
  int iter = 0;
  for (; iter < MAX_ITER; iter ++) {
    cilk::opadd_reducer<double> error = 0;
    cilk_for (vidType n = 0; n < nv; n ++)
      outgoing_contrib[n] = scores[n] / g.get_degree(n);
    cilk_for (vidType dst = 0; dst < nv; dst ++) {
      score_t incoming_total = 0;
      for (auto src : g.in_neigh(dst))
        incoming_total += outgoing_contrib[src];
      score_t old_score = scores[dst];
      scores[dst] = base_score + kDamp * incoming_total;
      error += fabs(scores[dst] - old_score);
    }
    printf(" %2d    %lf\n", iter+1, error);
    if (error < EPSILON) break;
  }
  std::cout << "iterations = " << iter+1 << ".\n";
  return;
}

