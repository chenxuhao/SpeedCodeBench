// optimized CUDA implementation
// one warp per each data point

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kmeans.h"
typedef unsigned long long AccType;

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define USE_SHM

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void assign_clusters_shm(
    const float* __restrict__ features,  // [npoints][dim]
    const float* __restrict__ centers,   // [nclusters][dim]
    int* __restrict__ membership,
    float* __restrict__ new_centers,     // [nclusters][dim]
    int* __restrict__ cluster_counts,
    int npoints, int dim, int nclusters, AccType* delta);
  
__global__ void assign_clusters(
    const float* __restrict__ features,  // [npoints][dim]
    const float* __restrict__ centers,   // [nclusters][dim]
    int* __restrict__ membership,
    float* __restrict__ new_centers,     // [nclusters][dim]
    int* __restrict__ cluster_counts,
    int npoints, int dim, int nclusters, AccType* delta);

void kmeans_clustering(int     dim,
                       int     npoints,
                       int     nclusters,
                       int64_t threshold,
                       float **features,    // in: [npoints][dim]
                       float **centroids,  // out: [nclusters][dim]
                       int    *memberships) // out: [npoints]
{
  size_t points_size = npoints * dim * sizeof(float);
  size_t centroids_size = nclusters * dim * sizeof(float);
  const float* h_features = features[0];
  float* h_centroids = centroids[0];
  float* d_features, *d_centroids, *d_new_centers;
  int* d_memberships, *d_cluster_sizes;
  AccType *d_delta;
  cudaMalloc(&d_delta, sizeof(AccType));
  cudaMalloc(&d_features, points_size);
  cudaMalloc(&d_centroids, centroids_size);
  cudaMalloc(&d_new_centers, centroids_size);
  cudaMalloc(&d_cluster_sizes, nclusters * sizeof(int));
  cudaMalloc(&d_memberships, npoints * sizeof(int));
  cudaMemcpy(d_features, h_features, points_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_memberships, memberships, npoints * sizeof(int), cudaMemcpyHostToDevice);
  float* h_new_centers = (float*)malloc(centroids_size);
  int* h_counts = (int*)malloc(nclusters * sizeof(int));
  AccType delta;

  int threads_per_block = BLOCK_SIZE;
  int num_blocks = (npoints * WARP_SIZE - 1) / threads_per_block + 1;
  //int num_blocks = (npoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t shm_bytes = nclusters * dim * sizeof(float) + nclusters * sizeof(int);
  printf("CUDA Kmeans (%d CTAs, %d threads/CTA)\n", num_blocks, threads_per_block );

  int loop = 0;
  do {
    delta = 0.;
    cudaMemcpy(d_delta, &delta, sizeof(AccType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice);
    cudaMemset(d_new_centers, 0, centroids_size);
    cudaMemset(d_cluster_sizes, 0, nclusters * sizeof(int));
#ifdef USE_SHM
    assign_clusters_shm<<<num_blocks, threads_per_block, shm_bytes>>>(d_features, d_centroids, d_memberships,
                                           d_new_centers, d_cluster_sizes, npoints, dim, nclusters, d_delta);
#else
    assign_clusters<<<num_blocks, threads_per_block>>>(d_features, d_centroids, d_memberships,
                            d_new_centers, d_cluster_sizes, npoints, dim, nclusters, d_delta);
#endif
    cudaDeviceSynchronize();
    cudaMemcpy(&delta, d_delta, sizeof(AccType), cudaMemcpyDeviceToHost);
    printf("iteration %d: delta=%ld\n", loop, delta);
    cudaMemcpy(h_new_centers, d_new_centers, centroids_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, d_cluster_sizes, nclusters * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nclusters; i++) {
      if (h_counts[i] > 0) {
        for (int j = 0; j < dim; j++) {
          h_centroids[i * dim + j] = h_new_centers[i * dim + j] / h_counts[i];
        }
      }
    }
    loop++;
  } while (delta > threshold && loop < 500);

  cudaMemcpy(memberships, d_memberships, npoints * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_features);
  cudaFree(d_centroids);
  cudaFree(d_new_centers);
  cudaFree(d_cluster_sizes);
  cudaFree(d_memberships);
}

__global__ void assign_clusters(
    const float* __restrict__ features,  // [npoints][dim]
    const float* __restrict__ centers,   // [nclusters][dim]
    int* __restrict__ membership,
    float* __restrict__ new_centers,     // [nclusters][dim]
    int* __restrict__ cluster_counts,
    int npoints, int dim, int nclusters,
    AccType* delta) {
    
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  if (warp_id >= npoints) return;

  const float* point = &features[warp_id * dim];
  float best_dist = INFINITY;
  int best_cluster = 0;

  for (int c = 0; c < nclusters; c++) {
    float partial = 0.0f;
    for (int j = lane; j < dim; j += WARP_SIZE) {
      float diff = point[j] - centers[c * dim + j];
      partial += diff * diff;
    }
    float dist = warp_reduce_sum(partial);

    if (lane == 0 && dist < best_dist) {
      best_dist = dist;
      best_cluster = c;
    }
  }

  // Only lane 0 writes the membership result
  if (lane == 0) {
    if (membership[warp_id] != best_cluster)
      atomicAdd(delta, 1ULL);
    membership[warp_id] = best_cluster;
    atomicAdd(&cluster_counts[best_cluster], 1);
  }

  // All lanes update new_centers collaboratively
  int cluster = __shfl_sync(0xffffffff, best_cluster, 0); // broadcast from lane 0
  for (int j = lane; j < dim; j += WARP_SIZE) {
    atomicAdd(&new_centers[cluster * dim + j], point[j]);
  }
}

extern __shared__ float shared_mem[];  // dynamic shared memory
// Layout: first nclusters * dim for new_centers, then nclusters for counts

__global__ void assign_clusters_shm(
    const float* __restrict__ features,  // [npoints][dim]
    const float* __restrict__ centers,   // [nclusters][dim]
    int* __restrict__ membership,
    float* __restrict__ new_centers,     // [nclusters][dim]
    int* __restrict__ cluster_counts,
    int npoints, int dim, int nclusters,
    AccType* delta) {

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int local_tid = threadIdx.x;

  // Shared memory setup
  float* smem_centers = (float*)shared_mem;              // nclusters * dim
  int* smem_counts = (int*)&smem_centers[nclusters * dim]; // nclusters

  // Zero out shared memory (one thread per cluster/feature)
  for (int i = local_tid; i < nclusters * dim; i += blockDim.x)
    smem_centers[i] = 0.0f;
  for (int i = local_tid; i < nclusters; i += blockDim.x)
    smem_counts[i] = 0;
  __syncthreads();

  // Bounds check
  if (warp_id >= npoints) return;

  const float* point = &features[warp_id * dim];
  float best_dist = INFINITY;
  int best_cluster = 0;

  // Warp-based Euclidean distance
  for (int c = 0; c < nclusters; c++) {
    float partial = 0.0f;
    for (int j = lane; j < dim; j += WARP_SIZE) {
      float diff = point[j] - centers[c * dim + j];
      partial += diff * diff;
    }
    float dist = warp_reduce_sum(partial);
    if (lane == 0 && dist < best_dist) {
      best_dist = dist;
      best_cluster = c;
    }
  }

  // Broadcast the best_cluster to all warp threads
  int cluster = __shfl_sync(0xffffffff, best_cluster, 0);

  if (lane == 0) {
    if (membership[warp_id] != cluster)
      atomicAdd(delta, 1ULL);
    membership[warp_id] = cluster;
    atomicAdd(&smem_counts[cluster], 1);
  }

  for (int j = lane; j < dim; j += WARP_SIZE) {
    atomicAdd(&smem_centers[cluster * dim + j], point[j]);
  }

  __syncthreads();

  // Now reduce shared memory to global memory (one thread block-wide)
  for (int i = local_tid; i < nclusters * dim; i += blockDim.x)
    atomicAdd(&new_centers[i], smem_centers[i]);
  for (int i = local_tid; i < nclusters; i += blockDim.x)
    atomicAdd(&cluster_counts[i], smem_counts[i]);
}

