#include "blackscholes.h"
#define BLOCK_SIZE 256
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)

__device__ inline float cndGPU(float d) {
  const float       A1 = 0.31938153f;
  const float       A2 = -0.356563782f;
  const float       A3 = 1.781477937f;
  const float       A4 = -1.821255978f;
  const float       A5 = 1.330274429f;
  const float RSQRT2PI = 0.39894228040143267793994605993438f;
  float K = 1.0f / (1.0f + 0.2316419f * fabsf(d));
  float cnd = RSQRT2PI * __expf(- 0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
  if (d > 0) cnd = 1.0f - cnd;
  return cnd;
}

__device__ inline void BlkSchlsEqEuroNoDiv(fptype sptprice,
                                           fptype strike,
                                           fptype rate,
                                           fptype volatility,
                                           fptype time,
                                           int otype) {
}

__global__ void BlackScholesGPU(
                 int numOptions, 
                 //OptionData* data,
                 int    * otype,
                 fptype * sptprice,
                 fptype * strike,
                 fptype * rate,
                 fptype * volatility,
                 fptype * otime,
                 fptype * prices) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numOptions) {
    BlkSchlsEqEuroNoDiv(sptprice[i],
                        strike[i],
                        rate[i],
                        volatility[i],
                        otime[i],
                        otype[i]);
  }
}

extern "C"
int BlackScholes(int numOptions, 
                 OptionData* data,
                 int    * otype,
                 fptype * sptprice,
                 fptype * strike,
                 fptype * rate,
                 fptype * volatility,
                 fptype * otime,
                 fptype * prices) {
  size_t OPT_SZ = numOptions * sizeof(fptype);
  int * d_otype;
  fptype *d_sptprice, *d_strike, *d_rate, *d_volatility, *d_otime, *d_prices;
  cudaMalloc((void **)&d_otype, numOptions * sizeof(int));
  cudaMalloc((void **)&d_sptprice, OPT_SZ);
  cudaMalloc((void **)&d_strike, OPT_SZ);
  cudaMalloc((void **)&d_rate, OPT_SZ);
  cudaMalloc((void **)&d_volatility, OPT_SZ);
  cudaMalloc((void **)&d_otime, OPT_SZ);
  cudaMalloc((void **)&d_prices, OPT_SZ);

  cudaMemcpy(d_sptprice, sptprice, OPT_SZ, cudaMemcpyHostToDevice);
  size_t nblocks = DIVIDE_INTO(numOptions, BLOCK_SIZE);

  int i;
  for (i = 0; i < NUM_RUNS; i++) {
    BlackScholesGPU<<<nblocks, BLOCK_SIZE>>>(numOptions, d_otype, d_sptprice, d_strike, d_rate, d_volatility, d_otime, d_prices);
  }
}
