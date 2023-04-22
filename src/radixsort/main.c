#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void radix_sort(unsigned int* out, unsigned int* const in, unsigned int d_in_len);
void verify(unsigned int* out, unsigned int* const in, unsigned int num_elems);
 
int main(int argc, char* argv[]) {
  unsigned int num_elems = (1 << 24);
  unsigned int* in       = (unsigned int*) malloc(sizeof(unsigned int)*num_elems);
  unsigned int* in_rand  = (unsigned int*) malloc(sizeof(unsigned int)*num_elems);
  unsigned int* out      = (unsigned int*) malloc(sizeof(unsigned int)*num_elems);
  if (argc > 1) num_elems = atoi(argv[1]);
  printf("array size: %d\n", num_elems);
  srand(1);
  for (unsigned int j = 0; j < num_elems; j++) {
    in[j] = (num_elems - 1) - j;
    in_rand[j] = rand() % num_elems;
  }
  for (int j = 0; j < 1; ++j) {
    printf("*****Descending order*****\n");
    double start = omp_get_wtime();
    radix_sort(out, in, num_elems);
    double end = omp_get_wtime();
    verify(out, in, num_elems);
    printf("running time: %f sec\n", end - start);

    printf("*****Random order*****\n");
    start = omp_get_wtime();
    radix_sort(out, in_rand, num_elems);
    end = omp_get_wtime();
    verify(out, in_rand, num_elems);
    printf("running time: %f sec\n", end - start);
  }
  free(out);
  free(in);
  free(in_rand);
}


