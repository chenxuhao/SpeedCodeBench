//#include <stdio.h>
//#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "timer.h"

// I/O routines
bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

void sgemm(char transa, char transb, 
           int m, int n, int k, 
           float alpha, const float *A, 
           int lda, const float *B, int ldb, 
           float beta, float *C, int ldc);
 
int main (int argc, char *argv[]) {
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;
  char* inpFiles[3];
  inpFiles[0] = argv[1];
  inpFiles[1] = argv[2];
  inpFiles[2] = argv[3];
  char* outFile = argv[4];

  // Read command line. Expect 3 inputs: A, B and B^T in column-major layout
  if ((inpFiles[0] == NULL) || (inpFiles[1] == NULL) || (inpFiles[2] == NULL)) {
    std::cerr << "Expecting three input filenames\n";
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

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v) {
  std::cerr << "Opening file:"<< fn << std::endl;
  std::fstream f(fn, std::fstream::in);
  if ( !f.good() ) {
    return false;
  }
  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;
  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  while (f.good() ) {
    f >> data;
    v.push_back(data);
  }
  v.pop_back(); // remove the duplicated last element
  return true;
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float>&v) {
  std::cerr << "Opening file:"<< fn << " for write." << std::endl;
  std::fstream f(fn, std::fstream::out);
  if ( !f.good() ) {
    return false;
  }
  // Read # of rows and cols
  f << nr_row << " "<<nr_col<<" ";
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  for (size_t i = 0; i < v.size(); ++i) {
    f << v[i] << ' ';
  }
  f << "\n";
  return true;
}
