#include <limits>
#include <cmath>
#include <stdlib.h>
#include "timer.h"
#include "graph.h"
static double DEFAULT_RELATIVE_TOL = 1e-4;
static double DEFAULT_ABSOLUTE_TOL = 1e-4;

template <typename T = float>
bool almost_equal(const T& a, const T& b, const double a_tol, const double r_tol) {
  using std::abs;
  if(abs(double(a - b)) > r_tol * (abs(double(a)) + abs(double(b))) + a_tol)
    return false;
  else
    return true;
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
bool check_almost_equal(const T * A, const T * B, const size_t N) {
  bool is_almost_equal = true;
  for(size_t i = 0; i < N; i++) {
    const T a = A[i];
    const T b = B[i];
    if(!almost_equal(a, b, DEFAULT_ABSOLUTE_TOL, DEFAULT_RELATIVE_TOL)) {
      is_almost_equal = false;
      printf("x_test[%ld] (%f) != x[%ld] (%f)\n", i, A[i], i, B[i]);
      break;
    }
  }
  return is_almost_equal;
}

template <typename T = float>
void gs_serial(eidType *Ap, vidType *Aj, vidType *indices, T *Ax, T *x, T *b, 
               int row_start, int row_stop, int row_step) {
  for (int i = row_start; i != row_stop; i += row_step) {
    auto inew = indices[i];
    auto row_begin = Ap[inew];
    auto row_end = Ap[inew+1];
    T rsum = 0;
    T diag = 0;
    for (auto jj = row_begin; jj < row_end; jj++) {
      auto j = Aj[jj];  //column index
      if (inew == j) diag = Ax[jj];
      else rsum += x[j] * Ax[jj];
    }
    if (diag != 0) x[inew] = (b[inew] - rsum) / diag;
  }
}

template <typename T = float>
void SymGSVerifier(GraphF &g, vidType *indices, T *test_x, T *x_host, T *b, std::vector<int> color_offsets) {
  printf("Verifying...\n");
  auto num_rows = g.V();
  auto Ap = g.in_rowptr();
  auto Aj = g.in_colidx();
  auto Ax = g.get_elabel_ptr();
  T *x = (T *)malloc(num_rows * sizeof(T));
  for (vidType i = 0; i < num_rows; i++)
    x[i] = x_host[i];
  Timer t;
  t.Start();
  for (size_t i = 0; i < color_offsets.size()-1; i++)
    gs_serial(Ap, Aj, indices, Ax, x, b, color_offsets[i], color_offsets[i+1], 1);
  for (size_t i = color_offsets.size()-1; i > 0; i--)
    gs_serial(Ap, Aj, indices, Ax, x, b, color_offsets[i-1], color_offsets[i], 1);
  t.Stop();
  printf("runtime [verify] = %f sec\n", t.Seconds());

  //for(int i = 0; i <10; i ++) printf("x_test[%d]=%f, x_ref[%d]=%f\n", i, test_x[i], i, x[i]);
  T max_error = maximum_relative_error<T>(test_x, x, num_rows);
  printf("[max error %9f]\n", max_error);
  //if ( max_error > 5 * std::sqrt( std::numeric_limits<T>::epsilon() ) )
  if(!check_almost_equal<T>(test_x, x, num_rows))
    printf("POSSIBLE FAILURE\n");
  else
    printf("Correct\n");
}
