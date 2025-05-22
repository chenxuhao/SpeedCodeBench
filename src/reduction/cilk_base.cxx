#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

extern "C"
int reduction(int n, int *arr, int *max, int *min) {
  int sum = 0;
  cilk::opadd_reducer<int> psum = 0;
  cilk_for (int i = 0; i < n; i++) {
    psum += arr[i];
  }
  sum = psum;

  int max_num = arr[0];
  int min_num = arr[0];
 
  for (int i = 1; i < n; i++) {
    // Check if the current number is greater than the current maximum value
    // Update the maximum value if the current number is greater
    if (arr[i] > max_num) max_num = arr[i];
    if (arr[i] < min_num) min_num = arr[i];
  }
  *max = max_num;
  *min = min_num;
 
  return sum;
}
