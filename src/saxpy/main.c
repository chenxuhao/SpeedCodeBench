//#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctimer.h>

#define NLUP  (32)
#define TWO26 (1 << 20)
void saxpy(const int n, const float a, const float *x, float *y);

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

  //#pragma omp parallel for default(none) shared(a, x, y, y_ref, y2, n) private(i)
  for (i = 0; i < n; ++i) {
    x[i]     = rand() % 32 / 32.0f;
    y[i]     = rand() % 32 / 32.0f;
    y_ref[i] = a * x[i] + y[i];
    y2[i] = 0.0f;
  }
  printf("total size of x and y is %f MB\n", 2.0 * nbytes / (1 << 20));
  memcpy(y2, y, nbytes);

  saxpy(n, a, x, y2);
  maxabserr = -1.0f;
  //#pragma omp parallel for reduction(max: maxabserr)
  for (i = 0; i < n; ++i) {
    float error =  fabsf(y2[i] - y_ref[i]);
    maxabserr = error > maxabserr? error : maxabserr;
  }

  saxpy(n, a, x, y2);

  ctimer_t t;
  ctimer_start(&t);
  //double start = omp_get_wtime();
  for (int i = 0; i < NLUP; ++i) {
    saxpy(n, a, x, y2);
  }
  //double end = omp_get_wtime();
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "saxpy");
}
