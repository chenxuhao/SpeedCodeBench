#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

typedef uint32_t data_t;
typedef uint32_t index_t;
#define BASE 8
#define BASE_BITS 3
#define DIGITS(x,y) (x >> y & 0x7)

extern "C"
void radix_sort(unsigned int* outdata,
                unsigned int* const indata,
                unsigned int N) {
  data_t *scratchdata = (data_t*) malloc(N * sizeof(data_t));
  data_t *src = indata, *dest = scratchdata;
  index_t total_digits = sizeof(data_t) * 8;
  index_t count[BASE], offset[BASE];
  int nthreads = __cilkrts_get_nworkers();
  index_t * local_counts = (index_t*) malloc(sizeof(index_t)*nthreads*BASE);
  index_t * local_offsets = (index_t*) malloc(sizeof(index_t)*nthreads*BASE);
  for (index_t shift = 0; shift < total_digits; shift+=BASE_BITS) {
    memset(count, 0, BASE*sizeof(index_t));
    memset(local_counts, 0, nthreads*BASE*sizeof(index_t));
    cilk_for (index_t i = 0; i < N; i++) {
      int tid = __cilkrts_get_worker_number();
      data_t key = DIGITS(src[i], shift);
      local_counts[tid*BASE+key]++;
    }
    for (int t = 0; t < nthreads; t++) {
      for (index_t b = 0; b < BASE; b++) {
        count[b] += local_counts[t*BASE+b];
      }
    }
    offset[0] = 0;
    for (index_t b = 1; b < BASE; b++)
      offset[b] = count[b-1] + offset[b-1];

    for (int t = 0; t < nthreads; t++) {
      for (index_t b = 0; b < BASE; b++) {
        local_offsets[t*BASE+b] = offset[b];
        offset[b] += local_counts[t*BASE+b];
      }
    }
    cilk_for (index_t i = 0; i < N; i++) {
      int tid = __cilkrts_get_worker_number();
      data_t key = DIGITS(src[i], shift);
      index_t pos = local_offsets[tid*BASE+key]++;
      dest[pos] = src[i];
    }
  }
  src = dest;
  dest = (dest == outdata) ? scratchdata : outdata;
}

