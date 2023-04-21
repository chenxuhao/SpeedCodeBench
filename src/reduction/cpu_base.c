#include <omp.h>
#include <stdio.h>

int reduction(int n, int *arr, int *max, int *min) {
  double start = omp_get_wtime();
  int i, sum = 0;
  for (i = 0; i < n; i++) {
    sum += arr[i];
  }
  double end = omp_get_wtime();
  printf("serial runtime = %f sec\n", end - start);
  return sum;
}
