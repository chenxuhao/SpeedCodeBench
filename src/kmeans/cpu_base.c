#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"

void kmeans_clustering(int     nfeatures,
                       int     npoints,
                       int     nclusters,
                       int64_t threshold,
                       float **feature,    // in: [npoints][nfeatures]
                       float **clusters,   // out: [nclusters][nfeatures]
                       int    *membership) // out: [npoints]
{
  // need to initialize new_centers_len and new_centers[0] to all 0
  float  **new_centers;     /* [nclusters][nfeatures] */
  int *new_centers_len = (int*) calloc(nclusters, sizeof(int));
  new_centers    = (float**) malloc(nclusters *            sizeof(float*));
  new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
  for (int i=1; i<nclusters; i++) new_centers[i] = new_centers[i-1] + nfeatures;
  int64_t delta = 0;
  int loop=0;
  do {
    delta = 0;
    for (int i=0; i<npoints; i++) {
      /* find the index of nestest cluster centers */
      int index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
      /* if membership changes, increase delta by 1 */
      if (membership[i] != index) delta += 1;
      /* assign the membership to object i */
      membership[i] = index;
      /* update new cluster centers : sum of objects located within */
      new_centers_len[index]++;
      for (int j=0; j<nfeatures; j++)          
        new_centers[index][j] += feature[i][j];
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
    //delta /= npoints;
    printf("iteration %d: delta=%ld\n", loop, delta);
  } while (delta > threshold && loop++ < 500);
  free(new_centers[0]);
  free(new_centers);
  free(new_centers_len);
}

