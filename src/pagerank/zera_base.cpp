#include <math.h>
#include "pr.h"
#include "BaseGraph.hh"
#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>
#include "../common/BaseGraph.cc"

void PRSolver(BaseGraph &g, score_t *_scores) {
  //int num_threads = __cilkrts_get_nworkers();
  //std::cout << "Cilk PangeRank (" << num_threads << " threads)\n";
  auto nv = g.V();
  const eidType* _verts = g.rowptr(); // get row pointers array
  const vidType* _edges = g.colidx(); // get column indices array
  auto rev_verts = g.in_rowptr();    // get row pointers array
  auto rev_edges = g.in_colidx();    // get column indices array
  std::vector<eidType>verts(_verts, _verts+nv+1);
  //std::vector<vidType>edges(_edges, _edges+g.E());
  std::vector<eidType>in_verts(rev_verts, rev_verts+nv+1);
  std::vector<vidType>in_edges(rev_edges, rev_edges+g.E());
 
  const score_t base_score = (1.0f - kDamp) / nv;
  std::vector<score_t> outgoing_contrib(nv);
  std::vector<score_t> scores(_scores, _scores+nv);

  int iter = 0;
  for (; iter < MAX_ITER; iter ++) {
    cilk::opadd_reducer<double> error = 0;
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (vidType n = 0; n < nv; n ++) {
      auto deg = verts[n+1] - verts[n];
      outgoing_contrib[n] = scores[n] / deg;
    }
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (vidType dst = 0; dst < nv; dst ++) {
      score_t incoming_total = 0;
      //for (auto src : g.in_neigh(dst))
      auto adj = &in_edges[in_verts[dst]];
      auto deg = vidType(in_verts[dst+1] - in_verts[dst]);
      for (vidType i = 0; i < deg; i++) {
        auto src = adj[i];
        incoming_total += outgoing_contrib[src];
      }
      score_t old_score = scores[dst];
      scores[dst] = base_score + kDamp * incoming_total;
      error += fabs(scores[dst] - old_score);
    }
    printf(" %2d    %lf\n", iter+1, error);
    if (error < EPSILON) break;
  }
  std::cout << "iterations = " << iter+1 << ".\n";
}

