#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

void convolution(float *Dst, float *Src, float *Kernel, int imageW, int imageH, int kernelR);
void verify(float *Dst, float *Src, float *Kernel, int imageW, int imageH, int kernelR);
 
int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]); 
    return 1;
  }
  const unsigned int imageW = atoi(argv[1]);
  const unsigned int imageH = atoi(argv[2]);
  const int   numIterations = atoi(argv[3]);

  float* h_Kernel = (float*)malloc(KERNEL_LENGTH   * sizeof(float));
  float* h_Input  = (float*)malloc(imageW * imageH * sizeof(float));
  float* h_Output = (float*)malloc(imageW * imageH * sizeof(float));

  srand(2009);
  for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    h_Kernel[i] = (float)(rand() % 8);
  for (unsigned int i = 0; i < imageW * imageH; i++)
    h_Input[i] = (float)(rand() % 8);

  double start = omp_get_wtime();
  for (int iter = 0; iter < numIterations; iter++)
    convolution(h_Output, h_Input, h_Kernel, imageW, imageH, imageW);
  double end = omp_get_wtime();
  printf("Average kernel execution time %f (s)\n", (end-start) / numIterations);

  verify(h_Output, h_Input, h_Kernel, imageW, imageH, imageW);

  free(h_Kernel);
  free(h_Output);
  free(h_Input);
  return 0;
}
