// Copyright (c) 2007 Intel Corp.
// Black-Scholes
// Analytical method for calculating European Options
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice  Hall, John C. Hull,

#include "blackscholes.h"

int BlackScholes(int numOptions, 
                 OptionData* data,
                 int    * otype,
                 fptype * sptprice,
                 fptype * strike,
                 fptype * rate,
                 fptype * volatility,
                 fptype * otime,
                 fptype * prices) {
  int numError = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP BlackSchole: %d threads\n", num_threads);
 
  int i, j;
  for (j = 0; j<NUM_RUNS; j++) {
    #pragma omp parallel for private(i)
    for (i = 0; i<numOptions; i++) {
      // Calling main function to calculate option value based on Black & Scholes's equation.
      fptype price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i], rate[i], volatility[i], otime[i], otype[i], 0);
      prices[i] = price;
      #ifdef ERR_CHK
      fptype priceDelta = data[i].DGrefval - price;
      if ( fabs(priceDelta) >= 1e-4 ) {
        printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n", i, price, data[i].DGrefval, priceDelta);
        numError ++;
      }
      #endif
    }
  }
  return numError;
}

