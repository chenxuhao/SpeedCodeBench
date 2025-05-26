#include <vector>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>
#include <opmax_reducer.h>
#include <opmin_reducer.h>
#include <algorithm>

extern "C"
int reduction(int n, int *arr, int *max, int *min) {
  std::vector<int> myarr(arr, arr + n);
  int sum = 0;
  cilk::opadd_reducer<int> psum = 0;
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int i = 0; i < n; i++) {
    psum += myarr[i];
  }
  sum = psum;

  //int max_num = arr[0];
  //int min_num = arr[0];
  cilk::opmin_reducer<int> pmin = arr[0];
  cilk::opmax_reducer<int> pmax = arr[0];
  //[[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int i = 1; i < n; i++) {
    pmin = std::min<int>(pmin, arr[i]);
    pmax = std::max<int>(pmax, arr[i]);
  }
  *max = pmax;
  *min = pmin;
  return sum;
}
