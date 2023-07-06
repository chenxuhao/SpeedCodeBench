#pragma once
#include <limits>
#include <cmath>
#include <algorithm>
#include "timer.h"

template <typename T = float>
inline size_t bytes_per_spmv(vidType m, eidType nnz) {
  size_t bytes = 0;
  bytes += 2*sizeof(T) * m;    // row pointer
  bytes += 1*sizeof(T) * nnz;  // column index
  bytes += 2*sizeof(T) * nnz;  // A[i,j] and x[j]
  bytes += 2*sizeof(T) * m;    // y[i] = y[i] + ...
  return bytes;
}

inline void print_throughput(int64_t m, int64_t nnz, double time) {
  assert(time > 0.);
  int64_t bytes = 12*nnz + 20*m;
  //printf("Bytes: %.2f \n", bytes);
  double GFLOPs = 2*double(nnz) / time / 10e9;
  double GBYTEs = double(bytes) / time / 10e9;
  std::cout << "Throughput: compute " << GFLOPs << " GFLOP/s, memory " << GBYTEs << " GB/s\n";
}

template <typename T = float>
T maximum_relative_error(const T * A, const T * B, const size_t N) {
  T max_error = 0;
  T eps = std::sqrt( std::numeric_limits<T>::epsilon() );
  for(size_t i = 0; i < N; i++) {
    const T a = A[i];
    const T b = B[i];
    const T error = std::abs(a - b);
    if (error != 0) {
      max_error = std::max(max_error, error/(std::abs(a) + std::abs(b) + eps) );
    }
  }
  return max_error;
}

template <typename T = float>
void SpmvSerial(vidType m, eidType nnz, const eidType *Ap, const vidType *Aj, const T *Ax, const T *x, T *y) {
  for (vidType i = 0; i < m; i++){
    auto row_begin = Ap[i];
    auto row_end   = Ap[i+1];
    auto sum = y[i];
    for (auto jj = row_begin; jj < row_end; jj++) {
      auto j = Aj[jj];  //column index
      sum += x[j] * Ax[jj];
    }
    y[i] = sum; 
  }
}

template <typename T = float>
T l2_error(size_t N, const T * a, const T * b) {
  T numerator   = 0;
  T denominator = 0;
  for (size_t i = 0; i < N; i++) {
    numerator   += (a[i] - b[i]) * (a[i] - b[i]);
    denominator += (b[i] * b[i]);
  }
  return numerator/denominator;
}

template <typename T = float>
void SpmvVerifier(GraphF &g, const T *x, T *y_test) {
  printf("Verifying...\n");
  auto m = g.V();
  auto nnz = g.E();
  auto Ap = g.in_rowptr();
  auto Aj = g.in_colidx();	
  auto Ax = g.get_elabel_ptr();
  std::vector<T> y(m, 0);
  Timer t;
  t.Start();
  SpmvSerial<T>(m, nnz, Ap, Aj, Ax, x, y.data());
  t.Stop();
  printf("runtime [serial] = %f s.\n", t.Seconds());
  auto max_error = maximum_relative_error(y_test, y.data(), m);
  printf("[max error %9f]\n", max_error);
  if ( max_error > 5 * std::sqrt( std::numeric_limits<T>::epsilon() ) )
    printf("POSSIBLE FAILURE\n");
  else
    printf("Correct\n");
}

