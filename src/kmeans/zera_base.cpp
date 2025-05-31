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

inline int find_nearest_point_(float*  pt,          /* [dim] */
                               int     dim,
                               float*  pts,         /* [npts][dim] */
                               int     npts) {
  int index = 0;
  float min_dist=FLT_MAX;
  for (int i=0; i<npts; i++) {
    float dist = euclid_dist_2(pt, &pts[i*dim], dim);
    if (dist < min_dist) {
      min_dist = dist;
      index    = i;
    }
  }
  return index;
}

extern "C"
void kmeans_clustering(int     dim,
                       int     npoints,
                       int     nclusters,
                       int64_t threshold,
                       float **_features,   // in: [npoints][dim]
                       float **_centers,    // out: [nclusters][dim]
                       int    *_membership) // out: [npoints]
{
  std::vector<int> membership(npoints, -1);
  FTP1D features(_features[0], _features[0]+npoints*dim);
  FTP1D centroids(_centers[0], _centers[0]+nclusters*dim);
  int nthreads = __cilkrts_get_nworkers();
  printf("Zera Histogram (%d threads)\n", nthreads);
  INT1D new_centers_len(nclusters, 0);
  FTP1D new_centers(dim*nclusters, 0);
  INT2D partial_new_centers_len(nthreads);
  for (int i=0; i<nthreads; i++) {
    partial_new_centers_len[i].resize(nclusters);
    std::fill(partial_new_centers_len[i].begin(), partial_new_centers_len[i].end(), 0);
  }
  FTP3D partial_new_centers(nthreads);
  for (int i=0; i<nthreads; i++) {
    partial_new_centers[i].resize(nclusters);
    for (int j=0; j<nclusters; j++) {
      partial_new_centers[i][j].resize(dim);
      std::fill(partial_new_centers[i][j].begin(), partial_new_centers[i][j].end(), 0.);
    }
  }

  int64_t delta = 0;
  int loop=0;
  do {
    cilk::opadd_reducer<int64_t> sum = 0;
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (int i=0; i<npoints; i++) {
      int tid = __cilkrts_get_worker_number();
      int index = find_nearest_point_(&features[i*dim], dim, &centroids[0], nclusters);
      if (membership[i] != index) sum += 1;
      membership[i] = index;
      partial_new_centers_len[tid][index]++;
      for (int j=0; j<dim; j++)
        partial_new_centers[tid][index][j] += features[i*dim+j];
    }
    delta = sum;

    /* let the main thread perform the array reduction */
    for (int i=0; i<nclusters; i++) {
      for (int j=0; j<nthreads; j++) {
        new_centers_len[i] += partial_new_centers_len[j][i];
        partial_new_centers_len[j][i] = 0.0;
        for (int k=0; k<dim; k++) {
          new_centers[i*dim+k] += partial_new_centers[j][i][k];
          partial_new_centers[j][i][k] = 0.0;
        }
      }
    }    
    for (int i=0; i<nclusters; i++) {
      for (int j=0; j<dim; j++) {
        if (new_centers_len[i] > 0)
          centroids[i*dim+j] = new_centers[i*dim+j] / new_centers_len[i];
        new_centers[i*dim+j] = 0.0;   /* set back to 0 */
      }
      new_centers_len[i] = 0;   /* set back to 0 */
    }
    printf("iteration %d: delta=%ld\n", loop, delta);
  } while (delta > threshold && loop++ < 500);
  printf("iterated %d times\n", loop);

  // copy results back
  for (int i=0; i<nclusters; i++)
    std::copy(centroids.begin(), centroids.end(), _centers[0]);
  std::copy(membership.begin(), membership.end(), _membership);
}

