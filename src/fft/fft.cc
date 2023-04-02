#include <math.h>
#include <stdio.h>
#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <inttypes.h>
#include "timer.h"

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

float randomFloat(float a, float b) {
  float random = ((float) rand()) / (float) RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

void inputData(char* fName, float* dat, int numwords) {
  if (fName == NULL) {
    for (int i = 0; i < numwords; i++) {
      dat[i] = randomFloat(1, 3);
    }
    return;
  }
  FILE* fid = fopen(fName, "r");
  if (fid == NULL) {
    fprintf(stderr, "Cannot open input file\n");
    exit(-1);
  }
  fread (dat, sizeof (float), numwords, fid);
  fclose (fid); 
}

void outputData(char* fName, float* outdat, int numwords) {
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;
  if (fid == NULL) {
    fprintf(stderr, "Cannot open output file\n");
    exit(-1);
  }
  /* Write the data size */
  tmp32 = numwords;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);
  /* Write the result data */
  fwrite (outdat, sizeof (float), numwords, fid);
  fclose (fid);
}

typedef struct float2 {
  float x;
  float y;
} float2;

float2 make_float2(float x, float y) {
  float2 result; result.x = x; result.y = y; return result;
}

inline float2 operator*( float2 a, float2 b ) { return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline float2 operator+( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }
inline float2 operator*( float2 a, float b ) { return make_float2( b*a.x , b*a.y); }

void compute_reference(float2 *dst, float2 *src, int batch, int n) {   
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
}   

int main( int argc, char **argv ) {	
  int n_bytes = 0, N = 10, B = 10;
  char* inpFile = NULL;
  char* outFile = NULL;
  char* numend;
  int err = 0;
  if (argc > 1) {
    N = strtol(argv[1], &numend, 10);
    if (numend == argv[1])
      err |= 2;
  }
  if (argc > 2) {
    B = strtol(argv[2], &numend, 10);
    if (numend == argv[2])
      err |= 4;
  }
  if (argc > 3) inpFile = argv[3];
  if (argc > 4) outFile = argv[4];
  printf("N = %d, B = %d\n", N, B);
  n_bytes = N*B*sizeof(float2);
  float2 *source = (float2 *)malloc( n_bytes );
  float2 *result = (float2 *)malloc( n_bytes );
  inputData(inpFile, (float*)source, N*B*2);
  Timer t;
  t.Start();
  compute_reference(result, source, B, N);	
  t.Stop();
  if (outFile) outputData(outFile, (float*)result, N*B*2);
  std::cout << "runtime [base] = " << t.Seconds() << " sec\n";
  free(source);
  free(result);
}

