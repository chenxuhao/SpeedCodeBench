#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);
extern int num_omp_threads;

int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */
                       int     npts);

float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{

  int      i, j, k, n=0, index, loop=0;
  int     *new_centers_len;			/* [nclusters]: no. of points in each cluster */
  float  **new_centers;				/* [nclusters][nfeatures] */
  float  **clusters;					/* out: [nclusters][nfeatures] */
  float    delta;
  int      nthreads;
  int    **partial_new_centers_len;
  float ***partial_new_centers;
  nthreads = num_omp_threads; 
  /* allocate space for returning variable clusters[] */
  clusters    = (float**) malloc(nclusters *             sizeof(float*));
  clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
  for (i=1; i<nclusters; i++)
    clusters[i] = clusters[i-1] + nfeatures;
  /* randomly pick cluster centers */
  for (i=0; i<nclusters; i++) {
    //n = (int)rand() % npoints;
    for (j=0; j<nfeatures; j++)
      clusters[i][j] = feature[n][j];
    n++;
  }
  for (i=0; i<npoints; i++)
    membership[i] = -1;

  /* need to initialize new_centers_len and new_centers[0] to all 0 */
  new_centers_len = (int*) calloc(nclusters, sizeof(int));
  new_centers    = (float**) malloc(nclusters *            sizeof(float*));
  new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
  for (i=1; i<nclusters; i++)
    new_centers[i] = new_centers[i-1] + nfeatures;
  partial_new_centers_len    = (int**) malloc(nthreads * sizeof(int*));
  partial_new_centers_len[0] = (int*)  calloc(nthreads*nclusters, sizeof(int));
  for (i=1; i<nthreads; i++)
    partial_new_centers_len[i] = partial_new_centers_len[i-1]+nclusters;
  partial_new_centers    =(float***)malloc(nthreads * sizeof(float**));
  partial_new_centers[0] =(float**) malloc(nthreads*nclusters * sizeof(float*));
  for (i=1; i<nthreads; i++)
    partial_new_centers[i] = partial_new_centers[i-1] + nclusters;
  for (i=0; i<nthreads; i++) {
    for (j=0; j<nclusters; j++)
      partial_new_centers[i][j] = (float*)calloc(nfeatures, sizeof(float));
  }
  printf("num of threads = %d\n", num_omp_threads);
  do {
    delta = 0.0;
    omp_set_num_threads(num_omp_threads);
#pragma omp parallel \
    shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len)
    {
      int tid = omp_get_thread_num();				
#pragma omp for \
      private(i,j,index) \
      firstprivate(npoints,nclusters,nfeatures) \
      schedule(static) \
      reduction(+:delta)
      for (i=0; i<npoints; i++) {
        /* find the index of nestest cluster centers */					
        index = find_nearest_point(feature[i],
            nfeatures,
            clusters,
            nclusters);				
        /* if membership changes, increase delta by 1 */
        if (membership[i] != index) delta += 1.0;

        /* assign the membership to object i */
        membership[i] = index;

        /* update new cluster centers : sum of all objects located
           within */
        partial_new_centers_len[tid][index]++;				
        for (j=0; j<nfeatures; j++)
          partial_new_centers[tid][index][j] += feature[i][j];
      }
    } /* end of #pragma omp parallel */

    /* let the main thread perform the array reduction */
    for (i=0; i<nclusters; i++) {
      for (j=0; j<nthreads; j++) {
        new_centers_len[i] += partial_new_centers_len[j][i];
        partial_new_centers_len[j][i] = 0.0;
        for (k=0; k<nfeatures; k++) {
          new_centers[i][k] += partial_new_centers[j][i][k];
          partial_new_centers[j][i][k] = 0.0;
        }
      }
    }    

    /* replace old cluster centers with new_centers */
    for (i=0; i<nclusters; i++) {
      for (j=0; j<nfeatures; j++) {
        if (new_centers_len[i] > 0)
          clusters[i][j] = new_centers[i][j] / new_centers_len[i];
        new_centers[i][j] = 0.0;   /* set back to 0 */
      }
      new_centers_len[i] = 0;   /* set back to 0 */
    }

  } while (delta > threshold && loop++ < 500);
  free(new_centers[0]);
  free(new_centers);
  free(new_centers_len);
  return clusters;
}

