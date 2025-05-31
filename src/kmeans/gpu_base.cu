// A naive CUDA implementation
// one CUDA thread per each data point: inefficient but easy to implement

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kmeans.h"

typedef unsigned long long AccType;

#define BLOCK_SIZE 256

inline __device__ float euclidean_distance(const float* p1, const float* p2, int dim) {
  float dist = 0.0f;
  for (int i = 0; i < dim; i++) {
    float d = p1[i] - p2[i];
    dist += d * d;
  }
  return dist;
}

__global__ void assign_clusters(
    const float* __restrict__ features,  // [npoints][dim]
    const float* __restrict__ centers,   // [nclusters][dim]
    int* __restrict__ membership,
    float* __restrict__ new_centers,     // [nclusters][dim]
    int* __restrict__ cluster_counts,
    int npoints, int dim, int nclusters,
    AccType *delta) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  const float* point = &features[tid * dim];
  int best_cluster = 0;
  float min_dist = euclidean_distance(point, &centers[0], dim);
  for (int c = 1; c < nclusters; c++) {
    float dist = euclidean_distance(point, &centers[c * dim], dim);
    if (dist < min_dist) {
      min_dist = dist;
      best_cluster = c;
    }
  }
  //printf("tid %d best_cluster %d\n", tid, best_cluster);
  if (membership[tid] != best_cluster) {
    atomicAdd(delta, 1.0);
    //printf("tid=%d: membership %d, delta %f\n", tid, membership[tid], delta);
  }
  membership[tid] = best_cluster;
  atomicAdd(&cluster_counts[best_cluster], 1);
  for (int j = 0; j < dim; j++) {
    atomicAdd(&new_centers[best_cluster * dim + j], point[j]);
  }
}

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
  int num_blocks = (npoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
  printf("CUDA Kmeans (%d CTAs, %d threads/CTA)\n", num_blocks, threads_per_block );

  int loop = 0;
  do {
    delta = 0.;
    cudaMemcpy(d_delta, &delta, sizeof(AccType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice);
    cudaMemset(d_new_centers, 0, centroids_size);
    cudaMemset(d_cluster_sizes, 0, nclusters * sizeof(int));
    assign_clusters<<<num_blocks, threads_per_block>>>(d_features, d_centroids, d_memberships,
                            d_new_centers, d_cluster_sizes, npoints, dim, nclusters, d_delta);
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

