#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <endian.h>
#include <malloc.h>
#include <inttypes.h>

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

void outputData(char* fName, float *h_A0, int nx, int ny, int nz) {
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;
  if (fid == NULL) {
    fprintf(stderr, "Cannot open output file\n");
    exit(-1);
  }
  tmp32 = nx*ny*nz;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);
  fwrite(h_A0, sizeof(float), tmp32, fid);
  fclose(fid);
}

static int read_data(float *A0, int nx,int ny,int nz,FILE *fp) {	
  int s = 0;
  int i, j, k;
  for (i=0; i<nz; i++) {
    for (j=0; j<ny; j++) {
      for (k=0; k<nx; k++) {
        fread(A0+s, sizeof(float), 1, fp);
        s++;
      }
    }
  }
  return 0;
}

void stencil(float c0, float c1, float *A0, float *Anext,
             const int nx, const int ny, const int nz, const int niter);

int main(int argc, char** argv) {
  int nx = 0, ny = 0, nz = 0;
  int size = 0;
  int iteration = 0;
  float c0 = 1.0f/6.0f;
  float c1 = 1.0f/6.0f/6.0f;
  if (argc < 5) {
    printf("Usage: probe nx ny nz tx ty t\n"
        "nx: the grid size x\n"
        "ny: the grid size y\n"
        "nz: the grid size z\n"
        "t: the iteration time\n");
    return -1;
  }
  nx = atoi(argv[1]);
  if (nx<1) return -1;
  ny = atoi(argv[2]);
  if (ny<1) return -1;
  nz = atoi(argv[3]);
  if (nz<1) return -1;
  iteration = atoi(argv[4]);
  if (iteration < 1) return -1;
  char* inpFile = argv[5];
  char* outFile = argv[6];
  printf("7 points Stencil with size %dx%dx%d and %d iterations\n", nx, ny, nz, iteration);

  float *h_A0;
  float *h_Anext;
  size = nx*ny*nz;
  h_A0 = (float*)malloc(sizeof(float)*size);
  h_Anext = (float*)malloc(sizeof(float)*size);
  FILE *fp = fopen(inpFile, "rb");
  read_data(h_A0, nx, ny, nz, fp);
  fclose(fp);
  memcpy(h_Anext, h_A0, sizeof(float) * size);
  stencil(c0, c1, h_A0, h_Anext, nx, ny, nz, iteration);
  if (outFile) outputData(outFile, h_Anext, nx, ny, nz);
  free (h_A0);
  free (h_Anext);
  return 0;
}
