#include <stdio.h>
#include <stdlib.h>
#include <ctimer.h>

#define SIZE 100000000
int reduction(int n, int *arr, int *max_num, int *min_num);

int main() {
  int *arr = malloc(SIZE * sizeof(int));

  // Initialize the array with random values
  srand(0);
  int i;
  for (i = 0; i < SIZE; i++) {
    arr[i] = rand() % 100;
  }
  int max = arr[0], min = arr[0];

  ctimer_t t;
  ctimer_start(&t);
  int sum = reduction(SIZE, arr, &max, &min);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "reduction");

  printf("The sum of the array is %d\n", sum);
  printf("The max of the array is %d\n", max);
  printf("The min of the array is %d\n", min);
  free(arr);
  return 0;
}

