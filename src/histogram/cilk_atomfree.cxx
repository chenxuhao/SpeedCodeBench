#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "opmin_reducer.h"
#include "opmax_reducer.h"
#include <vector>
#include <algorithm>

extern "C"
void histogram(unsigned int img_width,
               unsigned int img_height,
               unsigned int* image,
               unsigned int width,
               unsigned int height,
               unsigned char* histo) {
  int num_threads = __cilkrts_get_nworkers();
  printf("Cilk Histogram (%d threads)\n", num_threads);

  //calculate the max and min value of image[]
  cilk::opmin_reducer<int> pmin = image[0];
  cilk::opmax_reducer<int> pmax = image[0];
  cilk_for (int i = 1; i < img_width*img_height; i++) {
    pmin = std::min<int>(pmin, image[i]);
    pmax = std::max<int>(pmax, image[i]);
  }
  int max_val = pmax;
  int min_val = pmin;
  int num_histo = max_val - min_val + 1;
  std::vector<unsigned char> private_histo(num_threads*num_histo);
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

