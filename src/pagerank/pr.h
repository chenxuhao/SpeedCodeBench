#pragma once

// PageRank parameters
#define EPSILON 0.0001
#define MAX_ITER 100
const float kDamp = 0.85;
const float epsilon = 0.0000001;
const float epsilon2 = 0.001;

#include "math.h"
#include "graph.h"

/*
Kernel: PageRank (PR)
Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.

pr_omp_base: OpenMP implementation, one thread per vertex
pr_gpu_base: topology-driven GPU implementation using pull approach, one thread per vertex using CUDA
pr_gpu_push: topology-driven GPU implementation using push approach, one thread per edge using CUDA
*/

