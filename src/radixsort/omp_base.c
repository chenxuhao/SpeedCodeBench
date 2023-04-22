#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

typedef uint32_t data_t;
typedef uint32_t index_t;
#define BASE 8
#define BASE_BITS 3
#define DIGITS(x,y) (x >> y & 0x7)

void radix_sort(unsigned int* outdata,
                unsigned int* const indata,
                unsigned int N) {
  data_t *scratchdata = (data_t*) malloc(N * sizeof(data_t));
  data_t *src = indata, *dest = scratchdata;
  index_t total_digits = sizeof(data_t) * 8;
  index_t count[BASE], offset[BASE];
  for (index_t shift = 0; shift < total_digits; shift+=BASE_BITS) {
    memset(count, 0, BASE*sizeof(index_t));
    #pragma omp parallel
    {
      index_t local_count[BASE] = {0};
      index_t local_offset[BASE];
      #pragma omp for schedule(static) nowait
      for (index_t i = 0; i < N; i++) {
        data_t key = DIGITS(src[i], shift);
        local_count[key]++;
      }
      #pragma omp critical
      for (index_t b = 0; b < BASE; b++) {
        count[b] += local_count[b];
      }
      #pragma omp barrier
      #pragma omp single
      {
        offset[0] = 0;
        for (index_t b = 1; b < BASE; b++)
          offset[b] = count[b-1] + offset[b-1];
      }
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();
      for (int t = 0; t < nthreads; t++) {
        if (t == tid) {
          for (index_t b = 0; b < BASE; b++){
            local_offset[b] = offset[b];
            offset[b] += local_count[b];
          }
        }
        #pragma omp barrier
      }
      #pragma omp for schedule(static)
      for (index_t i = 0; i < N; i++) {
        data_t key = DIGITS(src[i], shift);
        index_t pos = local_offset[key]++;
        dest[pos] = src[i];
      }
    }
    src = dest;
    dest = (dest == outdata) ? scratchdata : outdata;
  }
}

