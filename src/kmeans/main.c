//   File:         example.c                                         
//   Description:  Takes as input a file:                              
//                 ascii  file: containing 1 data point per line       
//                 binary file: first int is the number of objects     
//                              2nd int is the no. of features of each 
//                              object                                 
//                 This example performs a fuzzy c-means clustering    
//                 on the data. Fuzzy clustering is performed using    
//                 min to max clusters and the clustering that gets    
//                 the best score according to a compactness and       
//                 separation criterion are returned.                  

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h> 
//#include "getopt.h"

extern double wtime(void);
int num_omp_threads = 1;

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

int cluster(int, int, float**, int, float, float***);

void usage(char *argv0) {
  char *help =
    "Usage: %s [switches] -i filename\n"
    "       -i filename     :  file containing data to be clustered\n"
    "       -b                 :input file is in binary format\n"
    "       -k                 : number of clusters (default is 8) \n"
    "       -t threshold    : threshold value\n";
  fprintf(stderr, help, argv0);
  exit(-1);
}

int main(int argc, char **argv) {
  int     opt;
  extern char   *optarg;
  //extern int     optind;
  int     nclusters=5;
  char   *filename = 0;           
  float  *buf;
  float **attributes;
  float **cluster_centres=NULL;
  int     i, j;           

  int     numAttributes;
  int     numObjects;           
  char    line[1024];
  int     isBinaryFile = 0;
  int     nloops;
  float   threshold = 0.001;
  double  timing;

  while ( (opt=getopt(argc,argv,"i:k:t:b"))!= EOF) {
    switch (opt) {
      case 'i': filename=optarg;
                break;
      case 'b': isBinaryFile = 1;
                break;
      case 't': threshold=atof(optarg);
                break;
      case 'k': nclusters = atoi(optarg);
                break;
      case '?': usage(argv[0]);
                break;
      default: usage(argv[0]);
               break;
    }
  }

  if (filename == 0) usage(argv[0]);
  numAttributes = numObjects = 0;

  /* from the input file, get the numAttributes and numObjects ------------*/
  if (isBinaryFile) {
    int infile;
    if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    if (read(infile, &numObjects, sizeof(int)) < 0)
      printf("WARNING: reading file error\n");
    if (read(infile, &numAttributes, sizeof(int)) < 0)
      printf("WARNING: reading file error\n");

    /* allocate space for attributes[] and read attributes of all objects */
    buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
    attributes    = (float**)malloc(numObjects*             sizeof(float*));
    attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
    for (i=1; i<numObjects; i++)
      attributes[i] = attributes[i-1] + numAttributes;
    if (read(infile, buf, numObjects*numAttributes*sizeof(float)) < 0)
      printf("WARNING: reading file error\n");
    close(infile);
  }
  else {
    FILE *infile;
    if ((infile = fopen(filename, "r")) == NULL) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
      if (strtok(line, " \t\n") != 0)
        numObjects++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") != 0) {
        /* ignore the id (first attribute): numAttributes = 1; */
        while (strtok(NULL, " ,\t\n") != NULL) numAttributes++;
        break;
      }
    }

    /* allocate space for attributes[] and read attributes of all objects */
    buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
    attributes    = (float**)malloc(numObjects*             sizeof(float*));
    attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
    for (i=1; i<numObjects; i++)
      attributes[i] = attributes[i-1] + numAttributes;
    rewind(infile);
    i = 0;
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") == NULL) continue; 
      for (j=0; j<numAttributes; j++) {
        buf[i] = atof(strtok(NULL, " ,\t\n"));
        i++;
      }
    }
    fclose(infile);
  }
  nloops = 1;	
  printf("I/O completed\n");
  memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));
  timing = omp_get_wtime();
  for (i=0; i<nloops; i++) {
    cluster_centres = NULL;
    cluster(numObjects,
        numAttributes,
        attributes,           /* [numObjects][numAttributes] */
        nclusters,
        threshold,
        &cluster_centres);
  }
  timing = omp_get_wtime() - timing;
  printf("number of Clusters %d\n",nclusters); 
  printf("number of Attributes %d\n\n",numAttributes); 
  /*printf("Cluster Centers Output\n"); 
    printf("The first number is cluster number and the following data is arribute value\n");
    printf("=============================================================================\n\n");

    for (i=0; i<nclusters; i++) {
    printf("%d: ", i);
    for (j=0; j<numAttributes; j++)
    printf("%f ", cluster_centres[i][j]);
    printf("\n\n");
    }*/
  printf("Time for process: %f\n", timing);
  free(attributes);
  free(cluster_centres[0]);
  free(cluster_centres);
  free(buf);
  return(0);
}

float **kmeans_clustering(float**, int, int, int, float, int*);

int cluster(int      numObjects,      /* number of input objects */
            int      numAttributes,   /* size of attribute of each object */
            float  **attributes,      /* [numObjects][numAttributes] */
            int      num_nclusters,
            float    threshold,       /* in:   */
            float ***cluster_centres /* out: [best_nclusters][numAttributes] */
            ) {
  int     nclusters;
  int    *membership;
  float **tmp_cluster_centres;
  membership = (int*) malloc(numObjects * sizeof(int));
  nclusters = num_nclusters;
  srand(7);
  tmp_cluster_centres = kmeans_clustering(attributes,
                                          numAttributes,
                                          numObjects,
                                          nclusters,
                                          threshold,
                                          membership);
  if (*cluster_centres) {
    free((*cluster_centres)[0]);
    free(*cluster_centres);
  }
  *cluster_centres = tmp_cluster_centres;
  free(membership);
  return 0;
}

/* multi-dimensional spatial Euclid distance square */
__inline float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    int i;
    float ans=0.0;
    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);
    return(ans);
}

int find_nearest_point(float  *pt,          /* [nfeatures] */
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

