#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>
#include "kmeans.h"

typedef std::vector<int>  INT1D;
typedef std::vector<float> FTP1D;
typedef std::vector<std::vector<int>>  INT2D;
typedef std::vector<std::vector<float>> FTP2D;
typedef std::vector<std::vector<std::vector<float>>> FTP3D;

inline int find_nearest_point_(FTP1D  &pt,          /* [nfeatures] */
                               int     nfeatures,
                               FTP2D  &pts,         /* [npts][nfeatures] */
                               int     npts) {
  int index = 0;
  float min_dist=FLT_MAX;
  /* find the cluster center id with min distance to pt */
  for (int i=0; i<npts; i++) {
    float dist;
    dist = euclid_dist_2(&pt[0], &pts[i][0], nfeatures);  /* no need square root */
    if (dist < min_dist) {
      min_dist = dist;
      index    = i;
    }
  }
  return index;
}

extern "C"
float** kmeans_clustering(float **_feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *_membership) /* out: [npoints] */
{
  std::vector<int> membership(npoints);
  for (int i=0; i<npoints; i++) membership[i] = -1;
  FTP2D feature;
  feature.reserve(npoints);
  for (int i=0; i<npoints; i++) {
    FTP1D feat(_feature[i], _feature[i] + nfeatures);
    feature.push_back(std::move(feat));
  }

  int nthreads = __cilkrts_get_nworkers();
  printf("Cilk Histogram (%d threads)\n", nthreads);

  FTP2D clusters(nclusters);
  for (int i=0; i<nclusters; i++) {
    clusters[i].resize(nfeatures);
    int n = i;
    for (int j=0; j<nfeatures; j++)
      clusters[i][j] = feature[n][j];
  }

  INT1D new_centers_len(nclusters, 0);
  FTP2D new_centers(nclusters);         // [nclusters][nfeatures]
  for (int i=0; i<nclusters; i++) {
    new_centers[i].resize(nfeatures);
    std::fill(new_centers[i].begin(), new_centers[i].end(), 0.);
  }
  INT2D partial_new_centers_len(nthreads);
  for (int i=0; i<nthreads; i++) {
    partial_new_centers_len[i].resize(nclusters);
    std::fill(partial_new_centers_len[i].begin(), partial_new_centers_len[i].end(), 0);
  }
  FTP3D partial_new_centers(nthreads);
  for (int i=0; i<nthreads; i++) {
    partial_new_centers[i].resize(nclusters);
    for (int j=0; j<nclusters; j++) {
      partial_new_centers[i][j].resize(nfeatures);
      std::fill(partial_new_centers[i][j].begin(), partial_new_centers[i][j].end(), 0.);
    }
  }

  float delta = 0.0;
  int loop=0;
  do {
    int tid = __cilkrts_get_worker_number();
    cilk::opadd_reducer<float> sum = 0.0;
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (int i=0; i<npoints; i++) {
      int index = find_nearest_point_(feature[i], nfeatures, clusters, nclusters);
      if (membership[i] != index) sum += 1.0;
      membership[i] = index;
      partial_new_centers_len[tid][index]++;
      for (int j=0; j<nfeatures; j++)
        partial_new_centers[tid][index][j] += feature[i][j];
    }
    delta = sum;

    /* let the main thread perform the array reduction */
    for (int i=0; i<nclusters; i++) {
      for (int j=0; j<nthreads; j++) {
        new_centers_len[i] += partial_new_centers_len[j][i];
        partial_new_centers_len[j][i] = 0.0;
        for (int k=0; k<nfeatures; k++) {
          new_centers[i][k] += partial_new_centers[j][i][k];
          partial_new_centers[j][i][k] = 0.0;
        }
      }
    }    

    /* replace old cluster centers with new_centers */
    for (int i=0; i<nclusters; i++) {
      for (int j=0; j<nfeatures; j++) {
        if (new_centers_len[i] > 0)
          clusters[i][j] = new_centers[i][j] / new_centers_len[i];
        new_centers[i][j] = 0.0;   /* set back to 0 */
      }
      new_centers_len[i] = 0;   /* set back to 0 */
    }
    printf("iteration %d: delta=%f\n", loop, delta);
  } while (delta > threshold && loop++ < 500);
  printf("iterated %d times\n", loop);

  // copy results back
  float  **_clusters;   /* out: [nclusters][nfeatures] */
  _clusters    = (float**) malloc(nclusters *             sizeof(float*));
  _clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
  for (int i=1; i<nclusters; i++) _clusters[i] = _clusters[i-1] + nfeatures;
  for (int i=0; i<nclusters; i++)
    std::copy(clusters[i].begin(), clusters[i].end(), _clusters[i]);
  std::copy(membership.begin(), membership.end(), _membership);
  return _clusters;
}

