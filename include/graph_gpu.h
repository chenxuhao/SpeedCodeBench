#pragma once
#include "BaseGraph.hh"
#include "common.h"
#include "cutil_subset.h"

class GraphGPU {
 protected:
  bool is_directed_;                // is it a directed graph?
  bool has_reverse;                 // has reverse/incoming edges maintained
  vidType num_vertices;             // number of vertices
  eidType num_edges;                // number of edges
  vidType max_degree;               // maximun degree
  eidType *d_rowptr, *d_in_rowptr;  // row pointers of CSR format
  vidType *d_colidx, *d_in_colidx;  // column induces of CSR format

 public:
  GraphGPU(BaseGraph &hg) : 
      GraphGPU(hg.V(), hg.E(), hg.is_directed(), hg.has_reverse_graph(), hg.get_max_degree()) {
    auto nv = hg.V();
    auto ne = hg.E();
    std::cout << "Copying graph to GPU\n";
    CUDA_SAFE_CALL(cudaMemcpy(d_rowptr, hg.rowptr(), (nv+1) * sizeof(eidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_colidx, hg.colidx(), ne * sizeof(vidType), cudaMemcpyHostToDevice));
    if (hg.has_reverse_graph()) {
      if (hg.is_directed()) {
        std::cout << "Copying reverse graph to GPU\n";
        CUDA_SAFE_CALL(cudaMemcpy(d_in_rowptr, hg.in_rowptr(), (nv+1) * sizeof(eidType), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_in_colidx, hg.in_colidx(), ne * sizeof(vidType), cudaMemcpyHostToDevice));
      }
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  GraphGPU(vidType nv, eidType ne, bool directed, bool reverse, vidType max_d) : 
      is_directed_(directed),
      has_reverse(reverse),
      num_vertices(nv),
      num_edges(ne),
      max_degree(max_d),
      d_rowptr(NULL),
      d_in_rowptr(NULL),
      d_colidx(NULL), 
      d_in_colidx(NULL) {
    assert (nv>0 && ne>0);
    allocateFrom(nv, ne, has_reverse);
  }
  void allocateFrom(vidType nv, eidType ne, bool has_reverse = false) {
    std::cout << "Allocating GPU memory for graph\n";
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr, (nv+1) * sizeof(eidType)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx, ne * sizeof(vidType)));
    if (has_reverse) {
      if (is_directed()) {
        std::cout << "Allocating GPU memory for reverse graph\n";
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_rowptr, (nv+1) * sizeof(eidType)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_colidx, ne * sizeof(vidType)));
      } else { // undirected graph; symmetric
        d_in_rowptr = d_rowptr;
        d_in_colidx = d_colidx;
      }
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  inline __device__ __host__ bool is_directed() { return is_directed_; }
  inline __device__ __host__ vidType V() { return num_vertices; }
  inline __device__ __host__ vidType size() { return num_vertices; }
  inline __device__ __host__ eidType E() { return num_edges; }
  inline __device__ __host__ eidType sizeEdges() { return num_edges; }
  inline __device__ __host__ vidType get_max_degree() { return max_degree; }
  inline __device__ __host__ bool valid_vertex(vidType vertex) { return (vertex < num_vertices); }
  inline __device__ __host__ bool valid_edge(eidType edge) { return (edge < num_edges); }
  inline __device__ __host__ vidType* N(vidType vid) { return d_colidx + d_rowptr[vid]; }
  inline __device__ __host__ vidType N(vidType v, eidType e) { return d_colidx[d_rowptr[v] + e]; }
  inline __device__ __host__ eidType* rowptr() { return d_rowptr; }
  inline __device__ __host__ vidType* colidx() { return d_colidx; }
  inline __device__ __host__ eidType* out_rowptr() { return d_rowptr; }
  inline __device__ __host__ vidType* out_colidx() { return d_colidx; }
  inline __device__ __host__ eidType* in_rowptr() { return d_in_rowptr; }
  inline __device__ __host__ vidType* in_colidx() { return d_in_colidx; }
  inline __device__ __host__ eidType getOutDegree(vidType src) { return d_rowptr[src+1] - d_rowptr[src]; }
  inline __device__ __host__ eidType getInDegree(vidType src) { return d_in_rowptr[src+1] - d_in_rowptr[src]; }
  inline __device__ __host__ vidType get_degree(vidType src) { return vidType(d_rowptr[src+1] - d_rowptr[src]); }
  inline __device__ __host__ vidType getEdgeDst(eidType edge) { return d_colidx[edge]; }
  inline __device__ __host__ vidType getOutEdgeDst(eidType edge) { return d_colidx[edge]; }
  inline __device__ __host__ vidType getInEdgeDst(eidType edge) { return d_in_colidx[edge]; }
  inline __device__ __host__ eidType edge_begin(vidType src) { return d_rowptr[src]; }
  inline __device__ __host__ eidType edge_end(vidType src) { return d_rowptr[src+1]; }
  inline __device__ __host__ eidType out_edge_begin(vidType src) { return d_rowptr[src]; }
  inline __device__ __host__ eidType out_edge_end(vidType src) { return d_rowptr[src+1]; }
  inline __device__ __host__ eidType in_edge_begin(vidType src) { return d_in_rowptr[src]; }
  inline __device__ __host__ eidType in_edge_end(vidType src) { return d_in_rowptr[src+1]; }
};

template <typename T = vidType>
__forceinline__ __device__ bool binary_search_2phase(T *list, T *cache, T key, T size) {
  int p = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  int mid = 0;
  // phase 1: search in the cache
  int bottom = 0;
  int top = WARP_SIZE;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    T y = cache[p + mid];
    if (key == y) return true;
    if (key < y) top = mid;
    if (key > y) bottom = mid;
  }

  //phase 2: search in global memory
  bottom = bottom * size / WARP_SIZE;
  top = top * size / WARP_SIZE - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    T y = list[mid];
    if (key == y) return true;
    if (key < y) top = mid - 1;
    else bottom = mid + 1;
  }
  return false;
}

template <typename T = vidType>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T* b, T size_b) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T cache[BLOCK_SIZE];
  T num = 0;
  T* lookup = a;
  T* search = b;
  T lookup_size = size_a;
  T search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i]; // each thread picks a vertex as the key
    if (binary_search_2phase(search, cache, key, search_size))
      num += 1;
  }
  return num;
}
