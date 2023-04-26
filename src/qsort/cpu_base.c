#include "ctimer.h"

void qsort(int* begin, int* end);

void QuickSort(int n, int *a) {
  ctimer_t t;
  ctimer_start(&t);
  qsort(a, a + n);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "qsort");
}

void swap(int* a, int* b) {
  int tmp = *a;
  *a = *b;
  *b = tmp;
}

// Partition array using last element of array as pivot
// (move elements less than last to lower partition
// and elements not less than last to upper partition
// return middle = the first element not less than last
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

// Sort the range between pointers begin and end.
// end is one past the final element in the range.
// Use the Quick Sort algorithm, using recursive divide and conquer.
void qsort(int* begin, int* end) {
  if (begin < end) {
    // get last element
    int last = *(end - 1);

    // we give partition a pointer to the first element and one past the last element
    // of the range we want to partition
    // move all values which are >= last to the end
    // move all value which are < last to the beginning
    // return a pointer to the first element >= last
    int * middle = partition(begin, end - 1, last);
    // move pivot to middle
    swap((end - 1), middle);
    qsort(middle+1, end); // sort upper partition w/o pivot
    qsort(begin, middle); // sort lower partition
  }
}
