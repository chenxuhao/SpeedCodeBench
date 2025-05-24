//#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctimer.h>
#include <assert.h>

#define NLUP  1
//#define TWO26 (1 << 25)
void saxpy(const int n, const float a, const float *x, float *y);

int main(int argc, char *argv[]) {
  int order = 25;
  if (argc > 1) order = atoi(argv[1]);
  assert(order > 10 & order < 30);
  int n = (1 << order);
  printf("SAXPY n = %d\n", n);

  ctimer_t t;
  ctimer_start(&t);
  float a = 2.0f, *x, *y, *y_ref, *y2, maxabserr;
  int64_t nbytes = sizeof(float) * n;
  x = (float *) malloc(nbytes);
  y = (float *) malloc(nbytes);
  y_ref = (float *) malloc(nbytes);
  y2 = (float *) malloc(nbytes);

  printf("check correctness\n");
  ctimer_start(&t);
  int i;
  for (i = 0; i < n; ++i) {
    x[i]     = rand() % 32 / 32.0f;
    y[i]     = rand() % 32 / 32.0f;
    y_ref[i] = a * x[i] + y[i];
    y2[i] = 0.0f;
  }
  //printf("total size of x and y is %f MB\n", 2.0 * nbytes / (1 << 20));
  memcpy(y2, y, nbytes);

  saxpy(n, a, x, y2);
  maxabserr = -1.0f;
  for (i = 0; i < n; ++i) {
    float error =  fabsf(y2[i] - y_ref[i]);
    maxabserr = error > maxabserr? error : maxabserr;
  }
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "correctness-check");

  printf("warmup\n");
  saxpy(n, a, x, y2);

  printf("start benchmarking\n");
  ctimer_start(&t);
  for (int i = 0; i < NLUP; ++i) {
    saxpy(n, a, x, y2);
  }
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "saxpy");
}
