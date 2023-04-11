// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "timer.h"
#include "graph_gpu.h"
#include "worklist.cuh"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
typedef Worklist2<vidType, vidType> WLGPU;

__global__ void first_fit_topo(GraphGPU g, int *colors, bool *changed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;	
  bool forbiddenColors[MAX_COLOR+1];
  if (colors[id] == MAX_COLOR) {
    for (int i = 0; i < MAX_COLOR; i++)
      forbiddenColors[i] = false;
    for (auto offset = g.edge_begin(id); offset < g.edge_end(id); offset ++) {
      auto neighbor = g.getEdgeDst(offset);
      int color = colors[neighbor];
      forbiddenColors[color] = true;
    }
    int vertex_color;
    for (vertex_color = 0; vertex_color < MAX_COLOR; vertex_color++) {
      if (!forbiddenColors[vertex_color]) {
        colors[id] = vertex_color;
        break;
      }
    }
    assert(vertex_color < MAX_COLOR);
    *changed = true;
  }
}

__global__ void conflict_resolve_topo(GraphGPU g, int *colors, bool *colored, WLGPU wl) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < g.V() && !colored[id]) {
    colored[id] = true;
    for (auto offset = g.edge_begin(id); offset < g.edge_end(id); offset ++) {
      auto neighbor = g.getEdgeDst(offset);
      if (id < neighbor && colors[id] == colors[neighbor]) {
        colors[id] = MAX_COLOR;
        colored[id] = false;
        wl.push(id);
        break;
      }
    }
  }
}

__global__ void first_fit(GraphGPU g, WLGPU inwl, int *colors) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;	
  bool forbiddenColors[MAX_COLOR+1];
  vidType vertex;
  if (inwl.pop_id(id, vertex)) {
    auto row_begin = g.edge_begin(vertex);
    auto row_end = g.edge_end(vertex+1);
    for (int j = 0; j < MAX_COLOR; j++)
      forbiddenColors[j] = false;
    for (auto offset = row_begin; offset < row_end; offset ++) {
      auto neighbor = g.getEdgeDst(offset);
      auto color = colors[neighbor];
      if(color != MAX_COLOR)
        forbiddenColors[color] = true;
    }
    int vertex_color;
    for (vertex_color = 0; vertex_color < MAX_COLOR; vertex_color++) {
      if (!forbiddenColors[vertex_color]) {
        colors[vertex] = vertex_color;
        break;
      }
    }
    assert(vertex_color < MAX_COLOR);
  }
}

__global__ void conflict_resolve(GraphGPU g, WLGPU inwl, WLGPU outwl, int *colors) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int conflicted = 0;
  vidType vertex;
  if (inwl.pop_id(id, vertex)) {
    auto row_begin = g.edge_begin(vertex);
    auto row_end = g.edge_end(vertex+1);
    for (auto offset = row_begin; offset < row_end; offset ++) {
      auto neighbor = g.getEdgeDst(offset);
      if (colors[vertex] == colors[neighbor] && vertex < neighbor) {
        conflicted = 1;
        colors[vertex] = MAX_COLOR;
        break;
      }
    }
  }
  if(conflicted) outwl.push(vertex);
}

int ColorSolver(Graph &g, int *colors) {
  auto m = g.V();
  GraphGPU gg(g);

  WLGPU inwl(m), outwl(m);
  WLGPU *inwlptr = &inwl, *outwlptr = &outwl;

  int *d_colors;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, m * sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_colors, colors, m * sizeof(int), cudaMemcpyHostToDevice));
  bool *d_changed, h_changed, *d_colored;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_colored, m * sizeof(bool)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (m - 1) / BLOCK_SIZE + 1;
  printf("CUDA Vertex Coloring (%d threads/CTA) ...\n", BLOCK_SIZE);

  int nitems = m;
  int num_colors = 0, iter = 0;
  first_fit_topo<<<nblocks, nthreads>>>(gg, d_colors, d_changed);
  conflict_resolve_topo<<<nblocks, nthreads>>>(gg, d_colors, d_colored, inwl);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
  while (nitems > 0) {
    iter ++;
    nblocks = (nitems - 1) / BLOCK_SIZE + 1;
    first_fit<<<nblocks, BLOCK_SIZE>>>(gg, inwl, d_colors);
    conflict_resolve<<<nblocks, BLOCK_SIZE>>>(gg, inwl, outwl, d_colors);
    nitems = outwlptr->nitems();
    WLGPU * tmp = inwlptr;
    inwlptr = outwlptr;
    outwlptr = tmp;
    outwlptr->reset();
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  CUDA_SAFE_CALL(cudaMemcpy(colors, d_colors, m * sizeof(int), cudaMemcpyDeviceToHost));
  #pragma omp parallel for reduction(max : num_colors)
  for (int n = 0; n < m; n ++)
    num_colors = max(num_colors, colors[n]);
  num_colors ++;
  printf("\titerations = %d.\n", iter);
  printf("\truntime [cuda_linear_base] = %f ms, num_colors = %d.\n", t.Millisecs(), num_colors);
  return num_colors;
}

