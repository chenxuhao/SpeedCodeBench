#include <omp.h>
#include <math.h>
#include <stdlib.h>

typedef unsigned InTy;
typedef unsigned OutTy;
typedef unsigned size_type;
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void prefix_sum(unsigned length, const InTy* in, OutTy *prefix) {
  const size_type block_size = 1 << 20;
  const size_type num_blocks = (length + block_size - 1) / block_size;
  OutTy* local_sums = (OutTy*)malloc(num_blocks*sizeof(OutTy));
  // count how many bits are set on each thread
  #pragma omp parallel for
  for (size_type block = 0; block < num_blocks; block ++) {
    OutTy lsum       = 0;
    size_type block_end = MIN((block + 1) * block_size, length);
    for (size_type i = block * block_size; i < block_end; i++)
      lsum += in[i];
    local_sums[block] = lsum;
  }
  OutTy* bulk_prefix = (OutTy*)malloc((num_blocks + 1)*sizeof(OutTy));
  OutTy total = 0;
  for (size_type block = 0; block < num_blocks; block++) {
    bulk_prefix[block] = total;
    total += local_sums[block];
  }
  bulk_prefix[num_blocks] = total;
  #pragma omp parallel for
  for (size_type block = 0; block < num_blocks; block ++) {
    OutTy local_total = bulk_prefix[block];
    size_type block_end  = MIN((block + 1) * block_size, length);
    for (size_type i = block * block_size; i < block_end; i++) {
      prefix[i] = local_total;
      local_total += in[i];
    }
  }
  prefix[length] = bulk_prefix[num_blocks];
}

