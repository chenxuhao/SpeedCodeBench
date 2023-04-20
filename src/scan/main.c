#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned T;
void prefix_sum(unsigned length, const T* in, T* out);

int main(int argc, char* argv[]) {
  unsigned n = 1024;
  if (argc > 1) n = atoi(argv[1]);
  T *input = NULL, *output = NULL;
  input = malloc(n*sizeof(T));
  for (unsigned i = 0; i < n; i++) {
    input[i] = rand() % 10;
  }
  output = malloc((n+1)*sizeof(T));
  double t0 = omp_get_wtime();
  prefix_sum(n, input, output);
  double t1 = omp_get_wtime();
  printf("Scan running time: %f\n", t1-t0);
  if (n < 10) {
    printf("input: ");
    for (unsigned i = 0; i < n; i++) {
      printf("%d ", input[i]);
    }
    printf("\noutput: ");
    for (unsigned i = 0; i < n+1; i++) {
      printf("%d ", output[i]);
    }
    printf("\n");
  }
  free(input);
  free(output);
}
