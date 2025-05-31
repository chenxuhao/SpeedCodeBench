#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float euclidean_distance_warp(const float* point, const float* cluster, int dim) {
    float local_sum = 0.0f;
    int lane = threadIdx.x % WARP_SIZE;
    for (int i = lane; i < dim; i += WARP_SIZE) {
        float diff = point[i] - cluster[i];
        local_sum += diff * diff;
    }
    return warp_reduce_sum(local_sum);
}

__global__ void assign_clusters(
    const float* __restrict__ features,   // [npoints][nfeatures]
    const float* __restrict__ clusters,   // [nclusters][nfeatures]
    int* __restrict__ membership,
    float* __restrict__ new_clusters,     // [nclusters][nfeatures]
    int* __restrict__ cluster_counts,
    int npoints, int nfeatures, int nclusters)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= npoints) return;

    int lane = threadIdx.x % WARP_SIZE;
    const float* point = &features[warp_id * nfeatures];

    float best_dist = INFINITY;
    int best_cluster = 0;

    for (int c = 0; c < nclusters; c++) {
        float dist = euclidean_distance_warp(point, &clusters[c * nfeatures], nfeatures);
        if (lane == 0 && dist < best_dist) {
            best_dist = dist;
            best_cluster = c;
        }
        __syncwarp();
    }

    if (lane == 0) {
        membership[warp_id] = best_cluster;
        atomicAdd(&cluster_counts[best_cluster], 1);
        for (int j = 0; j < nfeatures; j++) {
            atomicAdd(&new_clusters[best_cluster * nfeatures + j], point[j]);
        }
    }
}

float** kmeans_clustering(float **feature, int nfeatures, int npoints,
                          int nclusters, float threshold, int *membership) {
    size_t points_size = npoints * nfeatures * sizeof(float);
    size_t clusters_size = nclusters * nfeatures * sizeof(float);

    float* h_features = (float*)malloc(points_size);
    for (int i = 0; i < npoints; i++)
        for (int j = 0; j < nfeatures; j++)
            h_features[i * nfeatures + j] = feature[i][j];

    float* d_features, *d_clusters, *d_new_clusters;
    int* d_membership, *d_cluster_counts;

    cudaMalloc(&d_features, points_size);
    cudaMalloc(&d_clusters, clusters_size);
    cudaMalloc(&d_new_clusters, clusters_size);
    cudaMalloc(&d_cluster_counts, nclusters * sizeof(int));
    cudaMalloc(&d_membership, npoints * sizeof(int));

    cudaMemcpy(d_features, h_features, points_size, cudaMemcpyHostToDevice);

    float** clusters = (float**)malloc(nclusters * sizeof(float*));
    float* h_clusters = (float*)malloc(clusters_size);
    for (int i = 0; i < nclusters; i++) {
        clusters[i] = h_clusters + i * nfeatures;
        for (int j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[i][j];
    }

    float delta;
    int loop = 0;
    do {
        cudaMemcpy(d_clusters, h_clusters, clusters_size, cudaMemcpyHostToDevice);
        cudaMemset(d_new_clusters, 0, clusters_size);
        cudaMemset(d_cluster_counts, 0, nclusters * sizeof(int));

        int num_warps = (npoints + 1);
        int threads_per_block = BLOCK_SIZE;
        int blocks = (num_warps * WARP_SIZE + threads_per_block - 1) / threads_per_block;

        assign_clusters<<<blocks, threads_per_block>>>(d_features, d_clusters, d_membership,
                                                       d_new_clusters, d_cluster_counts,
                                                       npoints, nfeatures, nclusters);
        cudaDeviceSynchronize();

        float* h_new_clusters = (float*)malloc(clusters_size);
        int* h_counts = (int*)malloc(nclusters * sizeof(int));
        cudaMemcpy(h_new_clusters, d_new_clusters, clusters_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts, d_cluster_counts, nclusters * sizeof(int), cudaMemcpyDeviceToHost);

        delta = 0.0f;
        for (int i = 0; i < nclusters; i++) {
            if (h_counts[i] > 0) {
                for (int j = 0; j < nfeatures; j++) {
                    float new_val = h_new_clusters[i * nfeatures + j] / h_counts[i];
                    float old_val = clusters[i][j];
                    delta += (new_val - old_val) * (new_val - old_val);
                    clusters[i][j] = new_val;
                }
            }
        }
        printf("iteration %d: delta=%f\n", loop++, delta);
        free(h_new_clusters);
        free(h_counts);
    } while (delta > threshold && loop < 500);

    cudaFree(d_features);
    cudaFree(d_clusters);
    cudaFree(d_new_clusters);
    cudaFree(d_cluster_counts);
    cudaFree(d_membership);
    free(h_features);
    return clusters;
}

