#ifdef ENABLE_TBB
struct mainWork {
  mainWork(){}
  mainWork(mainWork &w, tbb::split){}

  void operator()(const tbb::blocked_range<int> &range) const {
    fptype price[NCO];
    fptype priceDelta;
    int begin = range.begin();
    int end = range.end();

    for (int i=begin; i!=end; i+=NCO) {
      /* Calling main function to calculate option value based on 
       * Black & Scholes's equation.
       */

      BlkSchlsEqEuroNoDiv( price, NCO, &(sptprice[i]), &(strike[i]),
                           &(rate[i]), &(volatility[i]), &(otime[i]), 
                           &(otype[i]), 0);
      for (int k=0; k<NCO; k++) {
        prices[i+k] = price[k];

#ifdef ERR_CHK 
        priceDelta = data[i+k].DGrefval - price[k];
        if( fabs(priceDelta) >= 1e-5 ){
          fprintf(stderr,"Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                 i+k, price, data[i+k].DGrefval, priceDelta);
          numError ++;
        }
#endif
      }
    }
  }
};

#endif // ENABLE_TBB

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_TBB
int bs_thread(void *tid_ptr) {
    int j;
    tbb::affinity_partitioner a;

    mainWork doall;
    for (j=0; j<NUM_RUNS; j++) {
      tbb::parallel_for(tbb::blocked_range<int>(0, numOptions), doall, a);
    }

    return 0;
}
#else // !ENABLE_TBB


