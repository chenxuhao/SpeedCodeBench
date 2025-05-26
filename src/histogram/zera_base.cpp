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

  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int i = 0; i < img_width*img_height; i++) {
    const unsigned int value = image[i];
    unsigned char old_val = histo[value];
    while (old_val < UINT8_MAX) {
      unsigned char new_val = old_val + 1;
      if (histo[value] == old_val && uchar_cas(&histo[value], old_val, new_val)) {
        break;
      }
      // Reload old_val to retry if CAS failed
      old_val = histo[value];
    }
  }
}

