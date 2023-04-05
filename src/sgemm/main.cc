#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include "timer.h"
#include "sgemm_base.cc"

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

int main (int argc, char *argv[]) {
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;
  char* inpFiles[3];
  inpFiles[0] = argv[1];
  inpFiles[1] = argv[2];
  inpFiles[2] = argv[3];
  char* outFile = argv[4];
  // take input filenames

  // Read command line. Expect 3 inputs: A, B and B^T in column-major layout
  if ((inpFiles[0] == NULL) || (inpFiles[1] == NULL) || (inpFiles[2] == NULL)) {
    fprintf(stderr, "Expecting three input filenames\n");
    exit(-1);
  }
 
  // load A
  readColMajorMatrixFile(inpFiles[0], matArow, matAcol, matA);
  // load B^T
  readColMajorMatrixFile(inpFiles[2], matBcol, matBrow, matBT);
  // allocate space for C
  std::vector<float> matC(matArow*matBcol);
  Timer t;
  t.Start();
  // Use standard sgemm interface
  sgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow,
        &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);

  if (outFile) {
    writeColMajorMatrixFile(outFile, matArow, matBcol, matC); 
  }
  t.Stop();
  double CPUtime = t.Seconds();
  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/CPUtime/1e9 << std::endl;
  return 0;
}
