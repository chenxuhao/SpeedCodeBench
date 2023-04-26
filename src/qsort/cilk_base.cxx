#include <cilk/cilk.h>

#include "ctimer.h"

void sample_qsort(int* begin, int* end);

extern "C"
void QuickSort(int n, int *a) {
  sample_qsort(a, a + n);
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

void sample_qsort(int* begin, int* end) {
  if (begin < end) {
    int last = *(end - 1);
    int * middle = partition(begin, end - 1, last);
    swap((end - 1), middle);
    cilk_scope {
      cilk_spawn sample_qsort(middle+1, end);
      sample_qsort(begin, middle);
    }
  }
}

