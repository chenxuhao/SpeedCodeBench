#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctimer.h>

void histogram(unsigned int img_width, unsigned int img_height, unsigned int* image,
               unsigned int width, unsigned int height, unsigned char* histo);
 
void dump_histo_img(unsigned char* histo, unsigned int height, unsigned int width, const char *filename);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("no input specified\n");
    return 1;
  }
  char* inpFile = argv[1];
  char* outFile = "out.bmp";
  if (argc > 2) outFile = argv[2];
  int numIterations = 1;
  if (argc > 3) numIterations = atoi(argv[1]);
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

  memset(histo, 0, histo_height*histo_width*sizeof(unsigned char));

  ctimer_t t;
  ctimer_start(&t);
  for (int i = 0; i < numIterations; i++) {
    histogram(img_width, img_height, img, histo_width, histo_height, histo);
  }
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "histogram");

  if (outFile) {
    printf("writing outputs to file %s\n", outFile);
    dump_histo_img(histo, histo_height, histo_width, outFile);
  }

  free(img);
  free(histo);
  return 0;
}
