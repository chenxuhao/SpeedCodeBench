#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//Merging 2 sorted subarrays into one tmp array
void merge(int * X, int n, int * tmp) {
  int i = 0;
  int j = n/2;
  int ti = 0;
  //i will iterate till first  half anf J will iterate for 2nd half of array
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
  #pragma omp task firstprivate (X, n, tmp)
  mergesort(X, n/2, tmp);
  #pragma omp task firstprivate (X, n, tmp)
  mergesort(X+(n/2), n-(n/2), tmp+n/2);
  #pragma omp taskwait
  merge(X, n, tmp);
}

void MergeSort(int n, int *data) {
  double start = omp_get_wtime();
  int *tmp = (int*) malloc(n*sizeof(int));
  #pragma omp parallel
  {
    #pragma omp single
    mergesort(data, n, tmp);
  }
  double stop = omp_get_wtime();
  printf("Time: %f sec\n", stop-start);
}
