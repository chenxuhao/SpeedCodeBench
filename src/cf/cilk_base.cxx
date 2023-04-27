// Copyright 2022
// Author: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

inline score_t rmse_par(int nv, int ne, score_t *errors) {
  cilk::opadd_reducer<score_t> tot_error = 0.0;
  cilk_for(int i = 0; i < nv; i ++) {
    tot_error += errors[i];
  }
  score_t total_error = sqrt(tot_error/ne);
  return total_error;
}

void SGDSolver(BipartiteGraph &g, std::vector<latent_t> &latents, int * ordering) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk CF (" << num_threads << " threads)\n";
  auto nv = g.V();
  std::vector<score_t> errors(nv*K, 0);
  #ifdef COMPUTE_ERROR
  std::vector<score_t> squared_errors(nv, 0.0);
  score_t total_error = 0.0;
  #endif
  int iter = 0;
  Timer t;
  t.Start();
  do {
    iter ++;
    #ifdef COMPUTE_ERROR
    for (vidType i = 0; i < nv; i ++) squared_errors[i] = 0;
    #endif
    cilk_for (vidType u = 0; u < nv; u ++) {
      latent_t *u_latent = &latents[K*u];
      latent_t *u_err = &errors[K*u];
      auto offset = g.edge_begin(u);
      for (auto v : g.N(u)) {
        latent_t *v_latent = &latents[K*v];
        score_t estimate = 0;
        for (int i = 0; i < K; i++)
          estimate += u_latent[i] * v_latent[i];
        score_t rating = g.getEdgeData(offset++);
        score_t delta = rating - estimate;
        #ifdef COMPUTE_ERROR
        squared_errors[u] += delta * delta;
        #endif
        for (int i = 0; i < K; i++)
          u_err[i] += v_latent[i] * delta;
      }
    }
    cilk_for (vidType u = 0; u < nv; u ++) {
      for (int i = 0; i < K; i++) {
        latents[K*u+i] += step * (-lambda * latents[K*u+i] + errors[K*u+i]);
        errors[K*u+i] = 0.0;
      }
    }
    #ifdef COMPUTE_ERROR
    total_error = rmse_par(nv, g.E(), &squared_errors[0]);
    printf("Iteration %d: RMSE error = %f\n", iter, total_error);
    if (total_error < cf_epsilon) break;
    #endif
  } while (iter < max_iters);
  t.Stop();
  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [cf_omp_base] = " << t.Seconds() << " sec\n";
  return;
}

