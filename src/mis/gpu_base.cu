// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "timer.h"
#include "graph_gpu.h"
#include "worklist.cuh"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
typedef Worklist2<vidType, vidType> WLGPU;

int ColorSolver(Graph &g, int *colors) {
  auto m = g.V();
  GraphGPU gg(g);

  // add your code to transfer data from CPU to GPU

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();

  // add your code to compute a kernel on GPU

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  // add your code to transfer data from GPU to CPU

}

