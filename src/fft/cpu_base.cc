#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "float2.h"

void fft(float2 *dst, float2 *src, int batch, int n) {   
  double start_time = omp_get_wtime();
  float2 *X = (float2*) malloc( n*sizeof(float2) );
  float2 *Y = (float2*) malloc( n*sizeof(float2) );
  for( int ibatch = 0; ibatch < batch; ibatch++ ) {
    // go to double precision
    for( int i = 0; i < n; i++ )
      X[i] = make_float2( src[i].x, src[i].y );
    // FFT in double precision
    for( int kmax = 1, jmax = n/2; kmax < n; kmax *= 2, jmax /= 2 ) {
      for( int k = 0; k < kmax; k++ ) {
        double phi = -2.*M_PI*k/(2.*kmax);
        float2 w = make_float2( cos(phi), sin(phi) ); 
        for( int j = 0; j < jmax; j++ ) {
          Y[j*2*kmax + k]        = X[j*kmax + k] + w * X[j*kmax + n/2 + k];
          Y[j*2*kmax + kmax + k] = X[j*kmax + k] - w * X[j*kmax + n/2 + k];
        }
      }
      float2 *Z = X;
      X = Y;
      Y = Z;
    }
    // return to single precision
    for( int i = 0; i < n; i++ )
      dst[i] = make_float2( (float)X[i].x, (float)X[i].y );
    src += n;
    dst += n;
  }
  free( X );
  free( Y );
  double end_time = omp_get_wtime();
  printf("runtime [base] = %f \n", end_time - start_time);
} 
