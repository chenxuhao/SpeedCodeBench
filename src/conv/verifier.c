#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void convolutionRowsHost(float *Dst, float *Src, float *Kernel,
                         int imageW, int imageH, int kernelR) {
  for (int y = 0; y < imageH; y++) {
    for (int x = 0; x < imageW; x++) {
      double sum = 0;
      for (int k = -kernelR; k <= kernelR; k++) {
        int d = x + k;
        if (d >= 0 && d < imageW)
          sum += Src[y * imageW + d] * Kernel[kernelR - k];
      }
      Dst[y * imageW + x] = (float)sum;
    }
  }
}

void convolutionColumnsHost(float *Dst, float *Src, float *Kernel,
                            int imageW, int imageH, int kernelR) {
  for (int y = 0; y < imageH; y++) {
    for (int x = 0; x < imageW; x++) {
      double sum = 0;
      for (int k = -kernelR; k <= kernelR; k++) {
        int d = y + k;
        if (d >= 0 && d < imageH)
          sum += Src[d * imageW + x] * Kernel[kernelR - k];
      }
      Dst[y * imageW + x] = (float)sum;
    }
  }
}
void verify(float *Dst, float *Src, float *Kernel, int imageW, int imageH, int kernelR) {
  float* Buffer = (float*) malloc(imageW * imageH * sizeof(float));
  float* Output = (float*) malloc(imageW * imageH * sizeof(float));
  printf("Comparing against Host/C++ computation...\n"); 
  convolutionRowsHost(Buffer, Src, Kernel, imageW, imageH, kernelR);
  convolutionColumnsHost(Output, Buffer, Kernel, imageW, imageH, kernelR);
  double sum = 0, delta = 0;
  for (unsigned int i = 0; i < imageW * imageH; i++) {
    delta += (Output[i] - Dst[i]) * (Output[i] - Dst[i]);
    sum += Output[i] * Output[i];
  }
  printf("sum=%f, delta=%f\n", sum, delta);
  double L2norm = sqrt(delta / sum);
  printf("Relative L2 norm: %.3e\n\n", L2norm);
  printf("%s\n", L2norm < 1e-6 ? "PASS" : "FAIL");
  free(Buffer);
  free(Output);
}
