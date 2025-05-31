//   Description:  Takes as input a file:                              
//                 ascii  file: containing 1 data point per line       
//                 binary file: first int is the number of objects 2nd    
//                              int is the no. of features of each object                                 
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
#include <unistd.h> 
//#include "getopt.h"
#include "ctimer.h"
#include "kmeans.h"

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
  char    line[1024];
  int     isBinaryFile = 0;
  float   threshold = 0.001;

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
  int dim = 0; // number of dimensions
  int npoints = 0; // number of data points
  float **attributes;

  /* from the input file, get the dim and npoints ------------*/
  if (isBinaryFile) {
    int infile;
    if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    if (read(infile, &npoints, sizeof(int)) < 0)
      printf("WARNING: reading file error\n");
    if (read(infile, &dim, sizeof(int)) < 0)
      printf("WARNING: reading file error\n");
    attributes    = (float**)malloc(npoints*    sizeof(float*));
    attributes[0] = (float*) malloc(npoints*dim*sizeof(float));
    for (int i=1; i<npoints; i++) attributes[i] = attributes[i-1] + dim;
    if (read(infile, attributes[0], npoints*dim*sizeof(float)) < 0)
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
        npoints++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") != 0) {
        /* ignore the id (first attribute): dim = 1; */
        while (strtok(NULL, " ,\t\n") != NULL) dim++;
        break;
      }
    }
    attributes    = (float**)malloc(npoints*    sizeof(float*));
    attributes[0] = (float*) malloc(npoints*dim*sizeof(float));
    for (int i=1; i<npoints; i++) attributes[i] = attributes[i-1] + dim;
    rewind(infile);
    int i = 0;
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") == NULL) continue; 
      for (int j=0; j<dim; j++) {
        attributes[0][i] = atof(strtok(NULL, " ,\t\n"));
        i++;
      }
    }
    fclose(infile);
  }
  //printf("I/O completed\n");

  // allocate space for returning variable memberships[]
  int *memberships = (int*) malloc(npoints * sizeof(int));
  for (int i=0; i<npoints; i++) memberships[i] = -1;

  // allocate space for returning variable clusters[]
  float **centroids;
  centroids    = (float**) malloc(nclusters *             sizeof(float*));
  centroids[0] = (float*)  malloc(nclusters * dim * sizeof(float));
  for (int i=1; i<nclusters; i++) centroids[i] = centroids[i-1] + dim;
 
  // randomly pick cluster centers
  for (int i=0; i<nclusters; i++) {
    int n = i;
    for (int j=0; j<dim; j++)
      centroids[i][j] = attributes[n][j];
  }

  printf("dim %d\n", dim);
  printf("number of Objects %d\n", npoints);
  printf("number of Clusters %d\n", nclusters); 
  //srand(7);

  ctimer_t t;
  ctimer_start(&t);
  kmeans_clustering(dim, npoints, nclusters, threshold, attributes, centroids, memberships);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "kmeans");

  //TODO: verify outputs
  //print_centroids(nclusters, dim, centroids);
  //print_memberships(npoints, memberships);

  free(memberships);
  free(centroids[0]);
  free(centroids);
  free(attributes[0]);
  free(attributes);
  return(0);
}

