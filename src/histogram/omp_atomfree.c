#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define UINT8_MAX 255

void histogram(unsigned int img_width,
               unsigned int img_height,
               unsigned int* image,
               unsigned int width,
               unsigned int height,
               unsigned char* histo) {
  //calculate the max and min value of image[]
  int min = image[0];
  int max = image[0];
  #pragma omp parallel for reduction(max: max) reduction(min: min)
  for (int i = 1; i < img_width*img_height; i++) {
    if (image[i] < min) min = image[i];
    if (image[i] > max) max = image[i];
  }
  int max_val = max;
  int min_val = min;

  int num_histo = max_val - min_val + 1;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP Histogram (%d threads)\n", num_threads);
  unsigned char * private_histo = (unsigned char*) calloc (num_threads*num_histo, sizeof(unsigned char));
 
  //accumulate the private histo
  #pragma omp for
  for (int i = 0; i < img_width*img_height; i++) {
    int tid = omp_get_thread_num();
    int index = tid*num_histo;
    private_histo[index+image[i]-min_val]++;
  }
  //combine the result into histo
  for (int j = min_val; j < max_val+1; j++) {
    for (int t = 0; t < num_threads; t++) {
      unsigned char temp = histo[j];
      histo[j] += private_histo[t*num_histo+j-min_val];
      if (histo[j] < temp) // uint8_t overflow
        histo[j] = UINT8_MAX;
    }
  }
}

