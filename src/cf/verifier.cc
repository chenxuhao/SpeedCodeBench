// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "cf.h"

// calculate RMSE
inline score_t rmse(vidType nv, eidType ne, score_t *errors) {
	score_t total_error = 0.0;
	for(vidType i = 0; i < nv; i ++)
		total_error += errors[i];
	total_error = sqrt(total_error/ne);
	return total_error;
}

void SGDVerifier(BipartiteGraph &g, std::vector<latent_t> &latents, int *ordering) {
  std::cout << "Verifying...\n";
  auto nv = g.V();
  //auto num_users = g.V(0);
  //auto num_items = g.V(1);
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
    for (vidType u = 0; u < nv; u ++) {
      latent_t *u_latent = &latents[K*u];
      latent_t *u_err = &errors[K*u];
      auto offset = g.edge_begin(u);
      for (auto v : g.N(u)) {
        latent_t *v_latent = &latents[K*v];
        score_t estimate = 0;
        for (int i = 0; i < K; i++) {
          estimate += u_latent[i] * v_latent[i];
        }
        score_t rating = g.getEdgeData(offset++);
        score_t delta = rating - estimate;
        #ifdef COMPUTE_ERROR
        squared_errors[u] += delta * delta;
        #endif
        for (int i = 0; i < K; i++) {
          u_err[i] += v_latent[i] * delta;
        }
      }
    }
    for (vidType u = 0; u < g.V(); u ++) {
      for (int i = 0; i < K; i++) {
        latents[K*u+i] += step * (-lambda * latents[K*u+i] + errors[K*u+i]);
        errors[K*u+i] = 0.0;
      }
    }
    #ifdef COMPUTE_ERROR
    total_error = rmse(nv, g.E(), &squared_errors[0]);
    printf("Iteration %d: RMSE error = %f\n", iter, total_error);
    if (total_error < cf_epsilon) break;
    #endif
  } while (iter < max_iters);
  t.Stop();
  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [verify] = " << t.Seconds() << " sec\n";
  return;
}

