//#pragma once
//#include <limits>
//#include <cmath>
//#include <algorithm>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <assert.h>
#include "ctimer.h"

typedef float T;
typedef uint32_t vidType;
typedef int64_t eidType;

#define EPSILON FLT_EPSILON

inline void print_throughput(int64_t m, int64_t nnz, double time) {
  assert(time > 0.);
  int64_t bytes = 12*nnz + 20*m;
  //printf("Bytes: %.2f \n", bytes);
  double GFLOPs = 2*(double)nnz / time / 10e9;
  double GBYTEs = (double)bytes / time / 10e9;
  printf("Throughput: compute %f GFLOP/s, memory %f GB/s\n", GFLOPs, GBYTEs);
}

inline T maximum_relative_error(const T * A, const T * B, const size_t N) {
  T max_error = 0;
  T eps = sqrt(EPSILON);
  for(size_t i = 0; i < N; i++) {
    const T a = A[i];
    const T b = B[i];
    const T error = abs(a - b);
    if (error != 0) {
      max_error = fmax(max_error, error/(abs(a) + abs(b) + eps) );
    }
  }
  return max_error;
}

inline T l2_error(size_t N, const T * a, const T * b) {
  T numerator   = 0;
  T denominator = 0;
  for (size_t i = 0; i < N; i++) {
    numerator   += (a[i] - b[i]) * (a[i] - b[i]);
    denominator += (b[i] * b[i]);
  }
  return numerator/denominator;
}

void SpmDmVerifier(char transa, char transb, 
                   vidType m, eidType nnz, int n,
                   T alpha, const eidType *Ap,
                   const vidType *Aj, const T *Ax, 
                   int lda, const T *BT, int ldb, 
                   T beta, T *C_test, int ldc) {
  printf("Verifying...\n");
  T* C = (T*) malloc(m*n*sizeof(T));
  for (vidType i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      T sum = 0;
      for (eidType off = Ap[i]; off < Ap[i+1]; off++) {
        vidType k = Aj[off];
        if (k >= m) printf("k=%u, off=%ld, i=%u, j=%d\n", k, off, i, j);
        assert(k < m);
        T value = Ax[off]; // A[i][k]
        sum += value * BT[j*m + k]; // A[i][k] * BT[j][k]
      }
      C[i*n + j] = sum; // C[i][j]
    }
  }
  T max_error = maximum_relative_error(C_test, C, m*n);
  printf("[max error %9f]\n", max_error);
  if ( max_error > 5 * sqrt(EPSILON) )
    printf("POSSIBLE FAILURE\n");
  else
    printf("Correct\n");
}

