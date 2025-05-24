#include <vector>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>
//#include <cilk/opmax_reducer.h>
//#include <cilk/opmin_reducer.h>

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

  int max_num = arr[0];
  int min_num = arr[0];
  /*
  cilk::reducer<cilk::op_max<int>> max_val(arr[0]);
  cilk::reducer<cilk::op_min<int>> min_val(arr[0]);
 
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int i = 1; i < n; i++) {
  */
  for (int i = 1; i < n; i++) {
    if (myarr[i] > max_num) max_num = myarr[i];
    if (myarr[i] < min_num) min_num = myarr[i];
  }
  *max = max_num;
  *min = min_num;
 
  return sum;
}
