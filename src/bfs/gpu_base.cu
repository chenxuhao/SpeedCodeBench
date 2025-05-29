#include "graph_gpu.h"
#include "worklist.cuh"
#include "ctimer.h"
#include "cuda_launch_config.hpp"

//#define MYINFINITY	1000000000
#define MYINFINITY	(-1)

__global__ void bfs_step(GraphGPU g, vidType *dists, WLGPU in_queue, WLGPU out_queue) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  vidType src;
  if (in_queue.pop_id(tid, src)) {
    //printf("src vid: %d\n", src);
    auto row_begin = g.edge_begin(src);
    auto row_end = g.edge_end(src);
    for (auto offset = row_begin; offset < row_end; ++ offset) {
      auto dst = g.getEdgeDst(offset);
      //printf("dst vid: %d\n", dst);
      if ((dists[dst] == MYINFINITY) && 
          (atomicCAS(&dists[dst], MYINFINITY, dists[src]+1) == MYINFINITY)) {
        out_queue.push(dst);
      }
    }
  }
}

__global__ void insert(vidType source, WLGPU queue) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id == 0) queue.push(source);
  return;
}

void BFSSolver(BaseGraph &g, vidType source, int *h_dists) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << double(memsize)/1024/1024/1024 << " GB, graph_mem = " << double(mem_graph)/1024/1024 << " MB\n";

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (nv-1)/nthreads+1;
  std::cout << "CUDA BFS (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  const vidType zero = 0;
  vidType * d_dists;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_dists, nv * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_dists, h_dists, nv * sizeof(vidType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(&d_dists[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  WLGPU queue1(nv), queue2(nv);
  WLGPU *in_frontier = &queue1, *out_frontier = &queue2;

  ctimer_t t;
  ctimer_start(&t);

  int iter = 0;
  int nitems = 1;
  insert<<<1, nthreads>>>(source, *in_frontier);
  nitems = in_frontier->nitems();
  printf("iteration %d: frontier_size = %d\n", iter, nitems);
  do {
    ++ iter;
    nblocks = (nitems - 1) / nthreads + 1;
    bfs_step<<<nblocks, nthreads>>>(gg, d_dists, *in_frontier, *out_frontier);
    nitems = out_frontier->nitems();
    WLGPU *tmp = in_frontier;
    in_frontier = out_frontier;
    out_frontier = tmp;
    out_frontier->reset();
    printf("iteration %d: frontier_size = %d\n", iter, nitems);
  } while (nitems > 0);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "BFS-gpu_base-kernel");

  double time = (double)timespec_nsec(t.elapsed) / 1e9;
  std::cout << "iterations = " << iter << ".\n";
  std::cout << "throughput = " << double(ne) / time / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
  CUDA_SAFE_CALL(cudaMemcpy(h_dists, d_dists, nv * sizeof(vidType), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_dists));
}

