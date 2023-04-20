// Copyright 2023 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

extern "C"
void saxpy(const int n, const float a, const float *x, float *y, const int) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk SAXPY (" << num_threads << " threads)\n"; 
  cilk_for (int i = 0; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
}
