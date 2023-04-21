#define UINT8_MAX 255

void histogram(unsigned int img_width,
               unsigned int img_height,
               unsigned int* image,
               unsigned int width,
               unsigned int height,
               unsigned char* histo) {
  for (int i = 0; i < img_width*img_height; ++i) {
    const unsigned int value = image[i];
    if (histo[value] < UINT8_MAX) {
      ++histo[value];
    }
  }
}
