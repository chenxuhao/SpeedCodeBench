#define ENABLE_SIMD
#include <immintrin.h>
#include "blackscholes.h"

#ifdef __GNUC__
#define _MM_ALIGN16 __attribute__((aligned (16)))
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#endif

// NCO = Number of Concurrent Options = SIMD Width
// NCO is currently set in the Makefile.
#ifndef NCO
#define NCO 4
#endif

#if (NCO==2)
#define SIMD_WIDTH 2
#define _MMR      __m128d
#define _MM_LOAD  _mm_load_pd
#define _MM_STORE _mm_store_pd
#define _MM_MUL   _mm_mul_pd
#define _MM_ADD   _mm_add_pd
#define _MM_SUB   _mm_sub_pd
#define _MM_DIV   _mm_div_pd
#define _MM_SQRT  _mm_sqrt_pd
#define _MM_SET(A)  _mm_set_pd(A,A)
#define _MM_SETR  _mm_set_pd
#endif

#if (NCO==4)
#define SIMD_WIDTH 4
#define _MMR      __m128
#define _MM_LOAD  _mm_load_ps
#define _MM_STORE _mm_store_ps
#define _MM_MUL   _mm_mul_ps
#define _MM_ADD   _mm_add_ps
#define _MM_SUB   _mm_sub_ps
#define _MM_DIV   _mm_div_ps
#define _MM_SQRT  _mm_sqrt_ps
#define _MM_SET(A)  _mm_set_ps(A,A,A,A)
#define _MM_SETR  _mm_set_ps
#endif

_MM_ALIGN16 OptionData* data;
_MM_ALIGN16 fptype* prices;

MUSTINLINE void CNDF ( fptype * OutputX, fptype * InputX ) {
  int sign[SIMD_WIDTH];
  int i;
  _MMR xInput;
  _MMR xNPrimeofX;
  _MM_ALIGN16 fptype expValues[SIMD_WIDTH];
  _MMR xK2;
  _MMR xK2_2, xK2_3, xK2_4, xK2_5;
  _MMR xLocal, xLocal_1, xLocal_2, xLocal_3;

  for (i=0; i<SIMD_WIDTH; i++) {
    // Check for negative value of InputX
    if (InputX[i] < 0.0) {
      InputX[i] = -InputX[i];
      sign[i] = 1;
    } else 
      sign[i] = 0;
  }
  // printf("InputX[0]=%lf\n", InputX[0]);
  // printf("InputX[1]=%lf\n", InputX[1]);

  xInput = _MM_LOAD(InputX);

  // local vars

  // Compute NPrimeX term common to both four & six decimal accuracy calcs

  for (i=0; i<SIMD_WIDTH; i++) {
    expValues[i] = exp(-0.5f * InputX[i] * InputX[i]);
    // printf("exp[%d]: %f\n", i, expValues[i]);
  }

  xNPrimeofX = _MM_LOAD(expValues);
  xNPrimeofX = _MM_MUL(xNPrimeofX, _MM_SET(inv_sqrt_2xPI));

  xK2 = _MM_MUL(_MM_SET(0.2316419), xInput);
  xK2 = _MM_ADD(xK2, _MM_SET(1.0));
  xK2 = _MM_DIV(_MM_SET(1.0), xK2);
  // xK2 = _mm_rcp_pd(xK2);  // No rcp function for double-precision

  xK2_2 = _MM_MUL(xK2, xK2);
  xK2_3 = _MM_MUL(xK2_2, xK2);
  xK2_4 = _MM_MUL(xK2_3, xK2);
  xK2_5 = _MM_MUL(xK2_4, xK2);

  xLocal_1 = _MM_MUL(xK2, _MM_SET(0.319381530));
  xLocal_2 = _MM_MUL(xK2_2, _MM_SET(-0.356563782));
  xLocal_3 = _MM_MUL(xK2_3, _MM_SET(1.781477937));
  xLocal_2 = _MM_ADD(xLocal_2, xLocal_3);
  xLocal_3 = _MM_MUL(xK2_4, _MM_SET(-1.821255978));
  xLocal_2 = _MM_ADD(xLocal_2, xLocal_3);
  xLocal_3 = _MM_MUL(xK2_5, _MM_SET(1.330274429));
  xLocal_2 = _MM_ADD(xLocal_2, xLocal_3);

  xLocal_1 = _MM_ADD(xLocal_2, xLocal_1);
  xLocal   = _MM_MUL(xLocal_1, xNPrimeofX);
  xLocal   = _MM_SUB(_MM_SET(1.0), xLocal);

  _MM_STORE(OutputX, xLocal);
  // _mm_storel_pd(&OutputX[0], xLocal);
  // _mm_storeh_pd(&OutputX[1], xLocal);

  for (i=0; i<SIMD_WIDTH; i++) {
    if (sign[i]) {
      OutputX[i] = (1.0 - OutputX[i]);
    }
  } 
} 

// For debugging
void print_xmm(_MMR in, char* s) {
  int i;
  _MM_ALIGN16 fptype val[SIMD_WIDTH];

  _MM_STORE(val, in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%f ", val[i]);
  }
  printf("\n");
}

void BlkSchlsEqEuroNoDiv (fptype * OptionPrice, fptype * sptprice,
                          fptype * strike, fptype * rate, fptype * volatility,
                          fptype * time, int * otype, float timet) {
  int i;
  // local private working variables for the calculation
  //_MMR xStockPrice;
  //_MMR xStrikePrice;
  _MMR xRiskFreeRate;
  _MMR xVolatility;
  _MMR xTime;
  _MMR xSqrtTime;

  _MM_ALIGN16 fptype logValues[NCO];
  _MMR xLogTerm;
  _MMR xD1, xD2;
  _MMR xPowerTerm;
  _MMR xDen;
  _MM_ALIGN16 fptype d1[SIMD_WIDTH];
  _MM_ALIGN16 fptype d2[SIMD_WIDTH];
  _MM_ALIGN16 fptype FutureValueX[SIMD_WIDTH];
  _MM_ALIGN16 fptype NofXd1[SIMD_WIDTH];
  _MM_ALIGN16 fptype NofXd2[SIMD_WIDTH];
  _MM_ALIGN16 fptype NegNofXd1[SIMD_WIDTH];
  _MM_ALIGN16 fptype NegNofXd2[SIMD_WIDTH];    

  //xStockPrice = _MM_LOAD(sptprice);
  //xStrikePrice = _MM_LOAD(strike);
  xRiskFreeRate = _MM_LOAD(rate);
  xVolatility = _MM_LOAD(volatility);
  xTime = _MM_LOAD(time);

  xSqrtTime = _MM_SQRT(xTime);

  for(i=0; i<SIMD_WIDTH;i ++) {
    logValues[i] = log(sptprice[i] / strike[i]);
  }

  xLogTerm = _MM_LOAD(logValues);

  xPowerTerm = _MM_MUL(xVolatility, xVolatility);
  xPowerTerm = _MM_MUL(xPowerTerm, _MM_SET(0.5));
  xD1 = _MM_ADD(xRiskFreeRate, xPowerTerm);

  xD1 = _MM_MUL(xD1, xTime);

  xD1 = _MM_ADD(xD1, xLogTerm);
  xDen = _MM_MUL(xVolatility, xSqrtTime);
  xD1 = _MM_DIV(xD1, xDen);
  xD2 = _MM_SUB(xD1, xDen);

  _MM_STORE(d1, xD1);
  _MM_STORE(d2, xD2);

  CNDF( NofXd1, d1 );
  CNDF( NofXd2, d2 );

  for (i=0; i<SIMD_WIDTH; i++) {
    FutureValueX[i] = strike[i] * (exp(-(rate[i])*(time[i])));
    // printf("FV=%lf\n", FutureValueX[i]);

    // NofXd1[i] = NofX(d1[i]);
    // NofXd2[i] = NofX(d2[i]);
    // printf("NofXd1=%lf\n", NofXd1[i]);
    // printf("NofXd2=%lf\n", NofXd2[i]);

    if (otype[i] == 0) {
      OptionPrice[i] = (sptprice[i] * NofXd1[i]) - (FutureValueX[i] * NofXd2[i]);
    }
    else { 
      NegNofXd1[i] = (1.0 - (NofXd1[i]));
      NegNofXd2[i] = (1.0 - (NofXd2[i]));
      OptionPrice[i] = (FutureValueX[i] * NegNofXd2[i]) - (sptprice[i] * NegNofXd1[i]);
    }
    // printf("OptionPrice[0] = %lf\n", OptionPrice[i]);
  }
}

