#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>
#include "opmin_reducer.h"
#include "opmax_reducer.h"
#include <algorithm>

extern "C"
int reduction(int n, int *arr, int *max, int *min) {
  int sum = 0;
  cilk::opadd_reducer<int> psum = arr[0];
  cilk_for (int i = 1; i < n; i++) {
    psum += arr[i];
  }
  sum = psum;

  cilk::opmin_reducer<int> pmin = arr[0];
  cilk::opmax_reducer<int> pmax = arr[0];
  cilk_for (int i = 1; i < n; i++) {
    pmin = std::min<int>(pmin, arr[i]);
    pmax = std::max<int>(pmax, arr[i]);
  }
  *min = pmin;
  *max = pmax;
 
  return sum;
}
