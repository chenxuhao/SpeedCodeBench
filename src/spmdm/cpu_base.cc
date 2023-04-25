// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "timer.h"
typedef float T;

// A: m x m 
// B: m x n
// BT:n x m
// C: m x n
void SpmDm(char transa, char transb, 
           vidType m, eidType nnz, int n,
           T alpha, const eidType *Ap,
           const vidType *Aj, const T *Ax, 
           int lda, const T *BT, int ldb, 
           T beta, T *C, int ldc) {
  printf("Serial SpMDM solver\n");
  Timer t;
  t.Start();
  for (vidType i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int sum = 0;
      for (int off = Ap[i]; off < Ap[i+1]; off++) {
        auto k = Aj[off];
        auto value = Ax[off]; // A[i][k]
        //sum += value * B[k*n + j]; // A[i][k] * B[k][j]
        sum += value * BT[j*m + k]; // A[i][k] * BT[j][k]
      }
      C[i*n + j] = sum; // C[i][j]
    }
  }
  t.Stop();
  double time = t.Seconds();
  assert(time > 0.);
  std::cout << "runtime [serial] = " << t.Seconds() << " sec\n";
  //float gbyte = bytes_per_spmdm(m, nnz) / 10e9;
  //float GFLOPs = 2*nnz / time / 10e9;
  //float GBYTEs = gbyte / time;
  //printf("Throughput: compute %5.2f GFLOP/s, memory %5.1f GB/s\n", GFLOPs, GBYTEs);
}

