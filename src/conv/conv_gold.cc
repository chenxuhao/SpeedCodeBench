void convolutionRowHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR) {
  for(int y = 0; y < imageH; y++) {
    for(int x = 0; x < imageW; x++) {
      double sum = 0;
      for(int k = -kernelR; k <= kernelR; k++){
        int d = x + k;
        if(d >= 0 && d < imageW)
          sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
      }
      h_Dst[y * imageW + x] = (float)sum;
    }
  }
}

void convolutionColumnHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR) {
  for(int y = 0; y < imageH; y++) {
    for(int x = 0; x < imageW; x++) {
      double sum = 0;
      for(int k = -kernelR; k <= kernelR; k++){
        int d = y + k;
        if(d >= 0 && d < imageH)
          sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
      }
      h_Dst[y * imageW + x] = (float)sum;
    }
  }
}
