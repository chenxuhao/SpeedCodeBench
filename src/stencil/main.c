#include <stdio.h>
#include <stdlib.h>
#include <cilk/cilk.h>

#define NX 100
#define NY 100
#define NZ 100

void stencil(double *u, double *v) {
  int i, j, k;
  double inv_denom = 1.0 / 7.0;

  cilk_for (k = 1; k < NZ-1; k++) {
    for (j = 1; j < NY-1; j++) {
      for (i = 1; i < NX-1; i++) {
        int index = i + NX * j + NX * NY * k;
        v[index] = (u[index-1] + u[index+1] + u[index-NX] + u[index+NX] +
            u[index-NX*NY] + u[index+NX*NY] + u[index]) * inv_denom;
      }
    }
  }
}

int main() {
  double *u, *v;
  int size = NX * NY * NZ;
  u = (double*)malloc(size * sizeof(double));
  v = (double*)malloc(size * sizeof(double));

  // Initialize u with some values
  for (int i = 0; i < size; i++) {
    u[i] = i;
  }

  stencil(u, v);

  // Print some values of v to verify correctness
  printf("v[0] = %f\n", v[0]);
  printf("v[10] = %f\n", v[10]);
  printf("v[100] = %f\n", v[100]);

  free(u);
  free(v);
  return 0;
}
