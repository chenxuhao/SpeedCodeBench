#include <omp.h>

int reduction(int n, int *arr, int *max, int *min) {
  int i, sum = 0;
  // Compute sum in parallel using OpenMP
  #pragma omp parallel for reduction(+: sum)
  for (i = 0; i < n; i++) {
    sum += arr[i];
  }

  int max_num = arr[0];
  int min_num = arr[0];
  // Find the maximum value in parallel using OpenMP
  #pragma omp parallel for reduction(max:max_num) reduction(min:min_num)
  for (i = 1; i < n; i++) {
    // Check if the current number is greater than the current maximum value
    // Update the maximum value if the current number is greater
    if (arr[i] > max_num) max_num = arr[i];
    if (arr[i] < min_num) min_num = arr[i];
  }
  *max = max_num;
  *min = min_num;
  return sum;
}
