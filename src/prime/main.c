# include <stdlib.h>
# include <stdio.h>

void prime_number_sweep ( int n_lo, int n_hi, int n_factor );

int main ( int argc, char *argv[] )
/*
  Purpose:

    MAIN is the main program for PRIME_OPENMP.

  Discussion:

    This program calls a version of PRIME_NUMBER that includes
    OpenMP directives for parallel processing.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 August 2009

  Author:

    John Burkardt
*/
{
  int n_factor;
  int n_hi;
  int n_lo;

  printf ( "\n" );
  printf ( "PRIME_OPENMP\n" );
  printf ( "  C/OpenMP version\n" );

  n_lo = 1;
  n_hi = 131072;
  n_factor = 2;

  prime_number_sweep ( n_lo, n_hi, n_factor );

  n_lo = 5;
  n_hi = 500000;
  n_factor = 10;

  prime_number_sweep ( n_lo, n_hi, n_factor );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "PRIME_OPENMP\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}
/******************************************************************************/


