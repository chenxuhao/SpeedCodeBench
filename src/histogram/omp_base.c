#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define UINT8_MAX 255

//calculate the min and max in parallel
void get_histo_size(unsigned int * img, int img_width, int img_height, int *max_val, int *min_val) {
  int min = img[0];
  int max = img[0];
  #pragma omp parallel for reduction(max: max) reduction(min: min)
  for (int i = 1; i < img_width*img_height; i++) {
    if (img[i] < min) min = img[i];
    if (img[i] > max) max = img[i];
  }
  *max_val = max;
  *min_val = min;
}

void histogram(unsigned int img_width,
               unsigned int img_height,
               unsigned int* image,
               unsigned int width,
               unsigned int height,
               unsigned char* histo) {
  //calculate the max and min value of image[]
  int max_val = 0;
  int min_val = 0;
  get_histo_size(image, img_width, img_height, &max_val, &min_val);
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

