#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cilk/cilk.h"
#include <cilk/cilk_api.h>
#include <vector>
#include "float2.h"

void fft(float2 *dst, float2 *src, int batch, int n) {
  int num_threads = __cilkrts_get_nworkers();
  printf("Cilk FFT solver (%d threads) ...\n", num_threads);
 
  std::vector<float2> X(n);
  std::vector<float2> Y(n);
  for (int ibatch = 0; ibatch < batch; ibatch++) {
    cilk_for (int i = 0; i < n; i++)
      X[i] = make_float2(src[i].x, src[i].y);

    // butterfly computation
    for (int kmax = 1, jmax = n/2; kmax < n; kmax *= 2, jmax /= 2) {
      [[tapir::target("cuda"), tapir::grain_size(1)]]
      //cilk_for (int k = 0; k < kmax; k++) {
      cilk_for (int index = 0; index < kmax * jmax; index++) {
        int k = index / jmax;
        int j = index % jmax;
        double phi = -2.*M_PI*k/(2.*kmax);
        float2 w = make_float2(cos(phi), sin(phi)); 
        //cilk_for (int j = 0; j < jmax; j++) {
          Y[j*2*kmax + k]        = X[j*kmax + k] + w * X[j*kmax + n/2 + k];
          Y[j*2*kmax + kmax + k] = X[j*kmax + k] - w * X[j*kmax + n/2 + k];
        //}
      }
      // swap pointers
      std::swap(X, Y);
    }

    cilk_for (int i = 0; i < n; i++)
      dst[i] = make_float2((float)X[i].x, (float)X[i].y);
    src += n;
    dst += n;
  }
}
