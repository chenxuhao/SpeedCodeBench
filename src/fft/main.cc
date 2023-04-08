#include <stdio.h>
#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <inttypes.h>
#include "float2.h"

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

void fft(float2 *dst, float2 *src, int batch, int n);
void inputData(char* fName, float* dat, int numwords);
void outputData(char* fName, float* outdat, int numwords);

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
  fft(result, source, B, N);
  if (outFile) outputData(outFile, (float*)result, N*B*2);
  free(source);
  free(result);
}

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
 
