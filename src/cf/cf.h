#pragma once

typedef float   score_t;   // score type for PageRank
typedef float   latent_t;  // latent type for CF

// CF parameters
const int K = 20;        // dimension of the latent vector (number of features)
extern float cf_epsilon; // convergence condition
extern score_t lambda;   // regularization_factor
extern score_t step;     // learning rate in the algorithm
extern int max_iters;    // maximum number of iterations

#include "graph.h"
#include <math.h>

