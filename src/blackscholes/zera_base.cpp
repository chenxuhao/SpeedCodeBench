#include <vector>
#include <cilk/cilk.h>
#include "blackscholes.h"
#include "blackscholes.c"

extern "C"
int BlackScholes(int n, 
                 OptionData* data_,
                 int    * otype_,
                 fptype * sptprice_,
                 fptype * strike_,
                 fptype * rate_,
                 fptype * volatility_,
                 fptype * otime_,
                 fptype * prices_) {
  std::vector<OptionData> data(data_, data_+n);
  std::vector<int> otype(otype_, otype_+n);
  std::vector<fptype> sptprice(sptprice_, sptprice_+n);
  std::vector<fptype> strike(strike_, strike_+n);
  std::vector<fptype> rate(rate_, rate_+n);
  std::vector<fptype> volatility(volatility_, volatility_+n);
  std::vector<fptype> otime(otime_, otime_+n);
  std::vector<fptype> prices(prices_, prices_+n);

  int numError = 0;
  for (int j = 0; j<NUM_RUNS; j++) {
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (int i = 0; i<n; i++) {
      // Calling main function to calculate option value based on Black & Scholes's equation.
      fptype price = BlkSchlsEqEuroNoDiv(sptprice[i], strike[i], rate[i], volatility[i], otime[i], otype[i], 0);
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

