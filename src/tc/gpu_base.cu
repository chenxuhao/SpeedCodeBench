#include <cub/cub.cuh>
#include "ctimer.h"
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

// vertex paralle: each warp takes one vertex
__global__ void triangle_bs_warp_vertex(vidType begin, vidType end, GraphGPU g, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  AccType count = 0;
  for (auto v = warp_id+begin; v < end; v += num_warps) {
    vidType *v_ptr = g.N(v);
    vidType v_size = g.getOutDegree(v);
    for (auto e = 0; e < v_size; e ++) {
      auto u = v_ptr[e];
      vidType u_size = g.getOutDegree(u);
      count += intersect_num(v_ptr, v_size, g.N(u), u_size);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if(threadIdx.x == 0) atomicAdd(total, block_num);
}

void TCSolver(BaseGraph &g, uint64_t &total) {
  //size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (nv-1)/WARPS_PER_BLOCK+1;
  //auto nnz = gg.init_edgelist(g);
  //size_t nblocks = (nnz-1)/WARPS_PER_BLOCK+1;

  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  //cudaProfilerStart();
  ctimer_t t;
  ctimer_start(&t);

  //triangle_bs_cta_vertex<<<nblocks, nthreads>>>(0, nv, gg, d_total);
  triangle_bs_warp_vertex<<<nblocks, nthreads>>>(0, nv, gg, d_total);
  //triangle_bs_cta_edge<<<nblocks, nthreads>>>(g.E(), gg, d_total);
  //triangle_bs_warp_edge<<<nblocks, nthreads>>>(g.E(), gg, d_total);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "BFS-gpu_base-kernel");
  double time = (double)timespec_nsec(t.elapsed) / 1e9;
  //cudaProfilerStop();

  std::cout << "throughput = " << double(g.E()) / time / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

