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
  int numAttributes = 0;
  int numObjects = 0;
  float *buf = NULL;
  float **attributes = NULL;

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
    for (int i=1; i<numObjects; i++)
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
    int i;
    for (i=1; i<numObjects; i++)
      attributes[i] = attributes[i-1] + numAttributes;
    rewind(infile);
    i = 0;
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") == NULL) continue; 
      for (int j=0; j<numAttributes; j++) {
        buf[i] = atof(strtok(NULL, " ,\t\n"));
        i++;
      }
    }
    fclose(infile);
  }
  //printf("I/O completed\n");

  memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));
  printf("number of Objects %d\n", numObjects);
  printf("number of Clusters %d\n",nclusters); 
  printf("number of Attributes %d\n",numAttributes); 

  ctimer_t t;
  ctimer_start(&t);
  int *membership = (int*) malloc(numObjects * sizeof(int));
  //srand(7);
  float **centres = kmeans_clustering(attributes,
                                        numAttributes,
                                        numObjects,
                                        nclusters,
                                        threshold,
                                        membership);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "kmeans");

  //print_centers(nclusters, numAttributes, centres);
  free(centres[0]);
  free(centres);
  free(membership);
  free(attributes);
  free(buf);
  return(0);
}

