#include "omp.h"
#include "stdio.h"

void qsort(int* begin, int* end);

void QuickSort(int n, int *a) {
  double start = omp_get_wtime();
  qsort(a, a + n);
  double end = omp_get_wtime();
  printf("Time %f sec\n", end - start);
}

void swap(int* a, int* b) {
  int tmp = *a;
  *a = *b;
  *b = tmp;
}

int* partition(int* begin, int* end, int pivot) {
  while (begin < end) {
    if (*begin < pivot) {
      begin++;
    } else {
      end--;
      swap(begin, end);
    }
  }
  return end;
}

void qsort(int* begin, int* end) {
  if (begin < end) {
    int last = *(end - 1);
    int * middle = partition(begin, end - 1, last);
    swap((end - 1), middle);
    #pragma omp task
    qsort(middle+1, end); // sort upper partition w/o pivot
    #pragma omp task
    qsort(begin, middle); // sort lower partition
    #pragma omp taskwait
  }
}
