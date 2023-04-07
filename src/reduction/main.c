#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 1000000

int main() {
  int sum = 0, i;
  int *arr = malloc(SIZE * sizeof(int));

  // Initialize the array with random values
  srand(0);
  for (i = 0; i < SIZE; i++) {
    arr[i] = rand() % 100;
  }

  // Compute sum in parallel using OpenMP
  #pragma omp parallel for reduction(+: sum)
  for (i = 0; i < SIZE; i++) {
    sum += arr[i];
  }

  int max_num = arr[0];
  // Find the maximum value in parallel using OpenMP
  #pragma omp parallel for reduction(max:max_num)
  for (int i = 1; i < SIZE; i++) {
    // Check if the current number is greater than the current maximum value
    if (arr[i] > max_num) {
      // Update the maximum value if the current number is greater
      max_num = arr[i];
    }
  }

  printf("The sum of the array is %d\n", sum);
  free(arr);
  return 0;
}

