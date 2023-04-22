#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define _0(x)   (x & 0x7FF)
#define _1(x)   (x >> 11 & 0x7FF)
#define _2(x)   (x >> 22 )

void radix_sort(unsigned int* sorted,
                unsigned int* const array,
                unsigned int elements) {
  // 3 histograms on the stack:
  const uint32_t kHist = 2048;
  uint32_t b0[kHist * 3];
  uint32_t *b1 = b0 + kHist;
  uint32_t *b2 = b1 + kHist;

  for (uint32_t i = 0; i < kHist * 3; i++)
    b0[i] = 0;
  //memset(b0, 0, kHist * 12);

  // 1.  parallel histogramming pass
  for (uint32_t i = 0; i < elements; i++) {
    uint32_t fi = array[i];
    b0[_0(fi)] ++;
    b1[_1(fi)] ++;
    b2[_2(fi)] ++;
  }
  // 2.  Sum the histograms -- each histogram entry records the number of values preceding itself.
  uint32_t sum0 = 0, sum1 = 0, sum2 = 0;
  uint32_t tsum;
  for (uint32_t i = 0; i < kHist; i++) {
    tsum = b0[i] + sum0;
    b0[i] = sum0 - 1;
    sum0 = tsum;
    tsum = b1[i] + sum1;
    b1[i] = sum1 - 1;
    sum1 = tsum;
    tsum = b2[i] + sum2;
    b2[i] = sum2 - 1;
    sum2 = tsum;
  }
  // byte 0: read/write histogram, write out
  for (uint32_t i = 0; i < elements; i++) {
    uint32_t fi = array[i];
    uint32_t pos = _0(fi);
    sorted[++b0[pos]] = fi;
  }
  // byte 1: read/write histogram, copy
  //   sorted -> array
  for (uint32_t i = 0; i < elements; i++) {
    uint32_t si = sorted[i];
    uint32_t pos = _1(si);
    array[++b1[pos]] = si;
  }
  // byte 2: read/write histogram, copy 
  //   array -> sorted
  for (uint32_t i = 0; i < elements; i++) {
    uint32_t ai = array[i];
    uint32_t pos = _2(ai);
    sorted[++b2[pos]] = ai;
  }
}
