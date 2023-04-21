#include <omp.h>
#include <stdlib.h>
 
void convolutionRows(float *Dst, float *Src, float *Kernel,
                     int imageW, int imageH, int kernelR) {
  #pragma omp parallel for collapse(2)
  for(int y = 0; y < imageH; y++) {
    for(int x = 0; x < imageW; x++) {
      double sum = 0;
      for(int k = -kernelR; k <= kernelR; k++){
        int d = x + k;
        if(d >= 0 && d < imageW)
          sum += Src[y * imageW + d] * Kernel[kernelR - k];
      }
      Dst[y * imageW + x] = (float)sum;
    }
  }
}

void convolutionColumns(float *Dst, float *Src, float *Kernel,
                        int imageW, int imageH, int kernelR) {
  #pragma omp parallel for collapse(2)
  for(int y = 0; y < imageH; y++) {
    for(int x = 0; x < imageW; x++) {
      double sum = 0;
      for(int k = -kernelR; k <= kernelR; k++){
        int d = y + k;
        if(d >= 0 && d < imageH)
          sum += Src[d * imageW + x] * Kernel[kernelR - k];
      }
      Dst[y * imageW + x] = (float)sum;
    }
  }
}

void convolution(float *Dst, float *Src, float *Kernel,
    int imageW, int imageH, int kernelR) {
  float* h_Buffer = (float*)malloc(imageW * imageH * sizeof(float));
  convolutionRows(h_Buffer, Src, Kernel, imageW, imageH, imageW);
  convolutionColumns(Dst, h_Buffer, Kernel, imageW, imageH, imageW);
  free(h_Buffer);
}
