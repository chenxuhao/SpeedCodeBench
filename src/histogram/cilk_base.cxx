#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

void get_histo_size(unsigned int * img, int img_width, int img_height, int *max_val, int *min_val);

extern "C"
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
  int num_threads = __cilkrts_get_nworkers();
  printf("Cilk Histogram (%d threads)\n", num_threads);
  unsigned char * private_histo = (unsigned char*) calloc(num_threads*num_histo, sizeof(unsigned char));
  //accumulate the private histo
  cilk_for (int i = 0; i < img_width*img_height; i++) {
    int tid = __cilkrts_get_worker_number();
    int index = tid*num_histo;
    private_histo[index+image[i]-min_val]++;
  }
  //combine the result into histo
  for (int j = min_val; j < max_val+1; j++) {
    for (int t = 0; t < num_threads; t++) {
      unsigned char temp = histo[j];
      histo[j] += private_histo[t*num_histo+j-min_val];
      if (histo[j] < temp) histo[j] = UINT8_MAX;
    }
  }
}

void zero_int(void *view) { *(int *)view = 0; }
void max_int(void *left, void *right) { *(int *)left = *(int *)right > *(int *)left ? *(int *)right : *(int *)left; }
void min_int(void *left, void *right) { *(int *)left = *(int *)right < *(int *)left ? *(int *)right : *(int *)left; }

//calculate the min and max values
void get_histo_size(unsigned int * img, int img_width, int img_height, int *max_val, int *min_val) {
  int cilk_reducer(zero_int, max_int) rmax;
  int cilk_reducer(zero_int, min_int) rmin;
  int num_threads = __cilkrts_get_nworkers();
  cilk_for (int i = 0; i < img_width*img_height; i++) {
    rmax = img[i] > rmax ? img[i] : rmax;
    rmin = img[i] < rmin ? img[i] : rmin;
  }
  *max_val = rmax;
  *min_val = rmin;
}

