#include "graph_gpu.h"

typedef int comp_t;

__global__ void hook(GraphGPU g, comp_t *comp, bool *changed) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < g.V()) {
		auto comp_src = comp[src];
		auto row_begin = g.edge_begin(src);
		auto row_end = g.edge_end(src); 
		for (auto offset = row_begin; offset < row_end; ++ offset) {
			auto dst = g.getEdgeDst(offset);
			auto comp_dst = comp[dst];
			if (comp_src == comp_dst) continue;
			auto high_comp = comp_src > comp_dst ? comp_src : comp_dst;
			auto low_comp = comp_src + (comp_dst - high_comp);
			if (high_comp == comp[high_comp]) {
				*changed = true;
				comp[high_comp] = low_comp;
			}
		}
	}
}

__global__ void shortcut(int m, comp_t *comp) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
		while (comp[src] != comp[comp[src]]) {
			comp[src] = comp[comp[src]];
		}
	}
}

void CCSolver(BaseGraph &g, comp_t *h_comp) {
  auto nv = g.num_vertices();
  GraphGPU gg(g);
  size_t nthreads = 256;
  size_t nblocks = (nv-1)/nthreads+1;
  std::cout << "CUDA CC (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
 
  comp_t *d_comp;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, sizeof(comp_t) * nv));
  CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, nv * sizeof(comp_t), cudaMemcpyHostToDevice));
  bool h_changed, *d_changed;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));

  int iter = 0;
  do {
    ++ iter;
    h_changed = false;
    CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
    hook<<<nblocks, nthreads>>>(gg, d_comp, d_changed);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    shortcut<<<nblocks, nthreads>>>(nv, d_comp);
    CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
  } while (h_changed);
  std::cout << "iterations = " << iter << ".\n";
  CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(comp_t) * nv, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_changed));
}

