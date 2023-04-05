#include <string.h>
#define UINT8_MAX 255

void histogram(int n,
               unsigned int img_width, unsigned int img_height,
               unsigned int* image,
               unsigned int width, unsigned int height,
               unsigned char* histo) {
  int iter;
  for (iter = 0; iter < n; iter++){
    memset(histo, 0, height*width*sizeof(unsigned char));
    unsigned int i;
    for (i = 0; i < img_width*img_height; ++i) {
      const unsigned int value = image[i];
      if (histo[value] < UINT8_MAX) {
        ++histo[value];
      }
    }
  }
}
