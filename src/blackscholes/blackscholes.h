#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

//Precision to use for calculations
#define fptype float

#define NUM_RUNS 100

// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

#ifdef ENABLE_SIMD
void CNDF ( fptype * OutputX, fptype * InputX );
void BlkSchlsEqEuroNoDiv (fptype * OptionPrice, fptype * sptprice,
                          fptype * strike, fptype * rate, fptype * volatility,
                          fptype * time, int * otype, float timet);
#else 
fptype CNDF ( fptype InputX );
fptype BlkSchlsEqEuroNoDiv( fptype sptprice,
                            fptype strike, fptype rate, fptype volatility,
                            fptype time, int otype, float timet );
#endif 

