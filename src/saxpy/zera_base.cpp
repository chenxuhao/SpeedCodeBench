#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>
//#include <stdio.h>

extern "C"
void saxpy(const int n, const float a, const float *x, float *y) {
  //int num_threads = __cilkrts_get_nworkers();
  //printf("Cilk GPU SAXPY (%d threads)\n", num_threads); 
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int i = 0; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
}
