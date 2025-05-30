#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>
#include "kmeans.h"

extern "C"
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{
  int     *new_centers_len;			/* [nclusters]: no. of points in each cluster */
  float  **new_centers;				/* [nclusters][nfeatures] */
  float  **clusters;					/* out: [nclusters][nfeatures] */
  int    **partial_new_centers_len;
  float ***partial_new_centers;
  int nthreads = __cilkrts_get_nworkers();
  printf("Cilk Histogram (%d threads)\n", nthreads);

  /* allocate space for returning variable clusters[] */
  clusters    = (float**) malloc(nclusters *             sizeof(float*));
  clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
  for (int i=1; i<nclusters; i++) clusters[i] = clusters[i-1] + nfeatures;
  /* randomly pick cluster centers */
  for (int i=0; i<nclusters; i++) {
    int n = i;
    for (int j=0; j<nfeatures; j++)
      clusters[i][j] = feature[n][j];
  }
  for (int i=0; i<npoints; i++) membership[i] = -1;

  /* need to initialize new_centers_len and new_centers[0] to all 0 */
  new_centers_len = (int*) calloc(nclusters, sizeof(int));
  new_centers    = (float**) malloc(nclusters *            sizeof(float*));
  new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
  for (int i=1; i<nclusters; i++)
    new_centers[i] = new_centers[i-1] + nfeatures;
  partial_new_centers_len    = (int**) malloc(nthreads * sizeof(int*));
  partial_new_centers_len[0] = (int*)  calloc(nthreads*nclusters, sizeof(int));
  for (int i=1; i<nthreads; i++)
    partial_new_centers_len[i] = partial_new_centers_len[i-1]+nclusters;
  partial_new_centers    =(float***)malloc(nthreads * sizeof(float**));
  partial_new_centers[0] =(float**) malloc(nthreads*nclusters * sizeof(float*));
  for (int i=1; i<nthreads; i++)
    partial_new_centers[i] = partial_new_centers[i-1] + nclusters;
  for (int i=0; i<nthreads; i++) {
    for (int j=0; j<nclusters; j++)
      partial_new_centers[i][j] = (float*)calloc(nfeatures, sizeof(float));
  }

  float delta = 0.0;
  int loop=0;
  do {
    int tid = __cilkrts_get_worker_number();
    cilk::opadd_reducer<float> sum = 0.0;
    cilk_for (int i=0; i<npoints; i++) {
      int index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);				
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
  free(new_centers[0]);
  free(new_centers);
  free(new_centers_len);
  return clusters;
}

