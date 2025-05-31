#pragma once

#include <stdio.h>
#include <stdlib.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifdef __cplusplus
extern "C" {
#endif
void kmeans_clustering(int, int, int, int64_t, float **, float **, int*);
#ifdef __cplusplus
}
#endif

/* multi-dimensional spatial Euclid distance square */
inline float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    int i;
    float ans=0.0;
    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);
    return(ans);
}

inline int find_nearest_point(float  *pt,          /* [nfeatures] */
                              int     nfeatures,
                              float **pts,         /* [npts][nfeatures] */
                              int     npts) {
  int index = 0, i = 0;
  float min_dist=FLT_MAX;
  /* find the cluster center id with min distance to pt */
  for (i=0; i<npts; i++) {
    float dist;
    dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
    if (dist < min_dist) {
      min_dist = dist;
      index    = i;
    }
  }
  return index;
}

inline void print_cnters(int nclusters, int numAttributes, float** cluster_centres) {
  printf("Cluster Centers Output\n"); 
  printf("The first number is cluster number and the following data is arribute value\n");
  for (int i=0; i<nclusters; i++) {
    printf("%d: ", i);
    for (int j=0; j<numAttributes; j++)
      printf("%f ", cluster_centres[i][j]);
    printf("\n\n");
  }
}

