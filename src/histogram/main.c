#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/******************************************************************************
* This implementations is a scalar, minimally optimized version. The only 
* optimization, which reduces the number of pointer chasing operations is the 
* use of a temporary pointer for each row.
******************************************************************************/

void histogram(int n,
               unsigned int img_width, unsigned int img_height, unsigned int* image,
               unsigned int width, unsigned int height, unsigned char* histo);
 
void dump_histo_img(unsigned char* histo, unsigned int height, unsigned int width, const char *filename);

int main(int argc, char* argv[]) {
  printf("Base implementation of histogramming.\n");
  int numIterations = 1;
  if (argc > 3) numIterations = atoi(argv[1]);
  char* inpFile = argv[1];
  char* outFile = argv[2];
  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;
  FILE* f = fopen(inpFile,"rb");
  int result = 0;
  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);
  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }
  printf("img_width=%u, img_height=%u, histo_width=%u, histo_height=%u \n",
         img_width, img_height, histo_width, histo_height);
  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));
  result = fread(img, sizeof(unsigned int), img_width*img_height, f);
  fclose(f);
  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }
  double start = omp_get_wtime();
  histogram(numIterations, img_width, img_height, img, histo_width, histo_height, histo);
  if (outFile) {
    printf("writing outputs to file %s\n", outFile);
    dump_histo_img(histo, histo_height, histo_width, outFile);
  }
  double end = omp_get_wtime();
  printf("runtime = %f sec\n", end - start);
  free(img);
  free(histo);
  return 0;
}
