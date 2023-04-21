//#include <cblas.h>

void saxpy(const int n, const float a, const float *x, float *y) {
  #pragma omp parallel for simd schedule(simd:static) default(none) shared(x, y)
  for (int i = 0; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
  //cblas_saxpy(n, a, x, 1, y, 1);
}
