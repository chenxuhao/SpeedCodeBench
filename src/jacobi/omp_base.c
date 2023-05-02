# include <omp.h>
# include <math.h>
# include <stdio.h>

// The linear system A*x=b is to be solved.
// Parameters:
//    Input,  int N, the order of the matrix.
//    Input,  double A[N,N], the matrix.
//    Input,  double B[N], the right hand side.
//    Input,  double X[N], the current solution estimate.
//    Output, double X[N], the solution estimate updated by
void jacobi(int m, int n, double *a, double *b, double *x0, double *x1) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP Jacobi solver (%d threads) ...\n", num_threads);
 
  double *x_old = x0;
  double *x_new = x1;
  double t0 = omp_get_wtime();
  for (int it = 0; it <= m; it++ ) {
    // one step of the Jacobi iteration.
    #pragma omp parallel for
    for (int i = 0; i < n; i++ ) {
      x_new[i] = b[i];
      for (int j = 0; j < n; j++ ) {
        if ( j != i ) {
          x_new[i] -= a[i+j*n] * x_old[j];
        }
      }
      x_new[i] /= a[i+i*n];
    }

    double * x = x_new;
    double r_norm = 0.0;
    // compute the norm of A*x-b.
    #pragma omp parallel for reduction(+:r_norm)
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
  double t1 = omp_get_wtime();
  printf("runtime [jacobi_omp_base] = %f \n", t1 - t0);
}

