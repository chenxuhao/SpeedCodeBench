#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Merging 2 sorted subarrays into one tmp array
void merge(int * X, int n, int * tmp) {
  int i = 0;
  int j = n/2;
  int ti = 0;
  while (i<n/2 && j<n) {
    if (X[i] < X[j]) {
      tmp[ti] = X[i];
      ti++; i++;
    } else {
      tmp[ti] = X[j];
      ti++;
      j++;
    }
  }
  while (i<n/2) { /* finish up lower half */
    tmp[ti] = X[i];
    ti++;
    i++;
  }
  while (j<n) { /* finish up upper half */
    tmp[ti] = X[j];
    ti++;
    j++;
  }
  //Copy sorted array tmp back to  X (Original array)
  memcpy(X, tmp, n*sizeof(int));
}

void mergesort(int * X, int n, int * tmp) {
  if (n < 2) return;
  mergesort(X, n/2, tmp);
  mergesort(X+(n/2), n-(n/2), tmp);
  merge(X, n, tmp);
}

void MergeSort(int n, int *data) {
  double start = omp_get_wtime();
  int *tmp = (int*) malloc(n*sizeof(int));
  mergesort(data, n, tmp);
  double stop = omp_get_wtime();
  printf("Time: %f sec\n", stop-start);
}
