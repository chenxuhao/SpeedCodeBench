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
               unsigned int* _image,
               unsigned int width,
               unsigned int height,
               unsigned char* histo) {
  int num_threads = __cilkrts_get_nworkers();
  printf("Cilk Histogram (%d threads)\n", num_threads);
  std::vector<unsigned int> image(_image, _image + img_width*img_height);

  //calculate the max and min value of image[]
  cilk::opmin_reducer<unsigned int> pmin = image[0];
  cilk::opmax_reducer<unsigned int> pmax = image[0];
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (unsigned int i = 1; i < img_width*img_height; i++) {
    pmin = std::min<unsigned int>(pmin, image[i]);
    pmax = std::max<unsigned int>(pmax, image[i]);
  }
  auto max_val = pmax;
  auto min_val = pmin;
  auto num_histo = max_val - min_val + 1;

  std::vector<unsigned char> private_histo(num_threads*num_histo);
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int i = 0; i < img_width*img_height; i++) {
    int tid = __cilkrts_get_worker_number();
    int index = tid*num_histo;
    private_histo[index+image[i]-min_val]++;
  }

  //combine per-thread results into global result
  for (int j = min_val; j < max_val+1; j++) {
    for (int t = 0; t < num_threads; t++) {
      auto temp = histo[j];
      histo[j] += private_histo[t*num_histo+j-min_val];
      if (histo[j] < temp) histo[j] = UINT8_MAX;
    }
  }
}

