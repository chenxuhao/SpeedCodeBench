#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NLUP  (32)
#define TWO26 (1 << 20)
void saxpy(const int n, const float a, const float *x, float *y, const int ial);

int main(int argc, char *argv[]) {
  int64_t nbytes = 0;
  float a = 2.0f, *x, *y, *y_ref, *y2, maxabserr;
  //int n = atoi(argv[1]);
  int n = TWO26, i = 0;
  nbytes = sizeof(float) * n;
  x = (float *) malloc(nbytes);
  y = (float *) malloc(nbytes);
  y_ref = (float *) malloc(nbytes);
  y2 = (float *) malloc(nbytes);

  #pragma omp parallel for default(none) shared(a, x, y, y_ref, y2, n) private(i)
  for (i = 0; i < n; ++i) {
    x[i]     = rand() % 32 / 32.0f;
    y[i]     = rand() % 32 / 32.0f;
    y_ref[i] = a * x[i] + y[i];
    y2[i] = 0.0f;
  }
  printf("total size of x and y is %9.1f MB\n", 2.0 * nbytes / (1 << 20));
  memcpy(y2, y, nbytes);

  saxpy(n, a, x, y2, 0);
  maxabserr = -1.0f;
  for (i = 0; i < n; ++i) {
    maxabserr = fabsf(y2[i] - y_ref[i]) > maxabserr?
      fabsf(y2[i] - y_ref[i]) : maxabserr;
  }

  saxpy(n, a, x, y2, 0);

  double start = omp_get_wtime();
  for (int i = 0; i < NLUP; ++i) {
    saxpy(n, a, x, y2, 0);
  }
  double end = omp_get_wtime();
  double wt = end - start;
  printf("saxpy: %9.1f MB/s maxabserr = %9.1f\n",
      NLUP * 3.0 * nbytes / ((1 << 20) * wt), maxabserr);
}
