#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>
//#include <stdio.h>
#include <vector>
#include <ctimer.h>

extern "C"
void saxpy(const int n, const float a, const float *x, float *y) {
  //int num_threads = __cilkrts_get_nworkers();
  //printf("Cilk GPU SAXPY (%d threads)\n", num_threads); 
  std::vector<float> myx(x, x + n);
  std::vector<float> myy(y, y + n);

  ctimer_t t;
  ctimer_start(&t);
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int i = 0; i < n; i++) {
    myy[i] = a * myx[i] + myy[i];
  }
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "saxpy-kernel");
}
