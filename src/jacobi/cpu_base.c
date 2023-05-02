# include <math.h>
# include <stdio.h>
# include "ctimer.h"

// The linear system A*x=b is to be solved.
// Parameters:
//    Input,  int N, the order of the matrix.
//    Input,  double A[N,N], the matrix.
//    Input,  double B[N], the right hand side.
//    Input,  double X[N], the current solution estimate.
//    Output, double X[N], the solution estimate updated by
void jacobi(int m, int n, double *a, double *b, double *x0, double *x1) {
  ctimer_t t;
  ctimer_start(&t);
 
  double *x_old = x0;
  double *x_new = x1;
  for (int it = 0; it <= m; it++ ) {
    // one step of the Jacobi iteration.
    for (int i = 0; i < n; i++ ) {
      x_new[i] = b[i];
      for (int j = 0; j < n; j++ ) {
        if ( j != i ) {
          x_new[i] = x_new[i] - a[i+j*n] * x_old[j];
        }
      }
      x_new[i] = x_new[i] / a[i+i*n];
    }

    double r_norm = 0.0;
    double * x = x_new;
    // compute the norm of A*x-b.
    for (int i = 0; i < n; i++ ) {
      double r = - b[i];
      for (int j = 0; j < n; j++ ) {
        r += a[i+j*n] * x[j];
      }
      r_norm += r * r;
    }
    r_norm = sqrt ( r_norm );
    if ( ( it <= 20 ) | ( ( it % 20 ) == 0 ) )
      printf ( "  %3d  %g\n", it, r_norm );

    // swap pointers
    x_new = x_old;
    x_old = x;
  }
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "jacobi_cpu_base");
}

