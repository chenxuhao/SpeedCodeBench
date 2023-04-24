#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>

void cpu_sort(int* out, int* in, size_t len);
void compare_results(int* out_test, int* const out, int n);
void MergeSort(int n, int *data);

void printArray(int arr[], int size) {
  for (int i = 0; i < size; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

int main(int argc, char* argv[]) {
  int num_elems = (1 << 24);
  int* in       = (int*) malloc(sizeof(int)*num_elems);
  int* in_rand  = (int*) malloc(sizeof(int)*num_elems);
  int* out      = (int*) malloc(sizeof(int)*num_elems);
  int* out_rand = (int*) malloc(sizeof(int)*num_elems);
  if (argc > 1) num_elems = atoi(argv[1]);
  printf("array size: %d\n", num_elems);
  srand(1);
  for (int j = 0; j < num_elems; j++) {
    in[j] = (num_elems - 1) - j;
    in_rand[j] = rand() % num_elems;
  }
  cpu_sort(out, in, num_elems);
  cpu_sort(out_rand, in_rand, num_elems);

  for (int j = 0; j < 1; ++j) {
    printf("*****Descending order*****\n");
    MergeSort(num_elems, in);
    compare_results(in, out, num_elems);

    printf("*****Random order*****\n");
    //start = omp_get_wtime();
    MergeSort(num_elems, in_rand);
    //end = omp_get_wtime();
    compare_results(in_rand, out_rand, num_elems);
  }
  free(in);
  free(in_rand);
  free(out);
  free(out_rand);
  return 0;
}

// Check for any mismatches between outputs of MergeSort and std::sort
void compare_results(int* out_test, int* const h_out_cpu, int num_elems) {
  bool match = true;
  int index_diff = 0;
  for (unsigned int i = 0; i < num_elems; ++i) {
    if (h_out_cpu[i] != out_test[i]) {
      match = false;
      index_diff = i;
      break;
    }
  }
  if (match) printf("PASS\n");
  else printf("Mismatch!\n");

  // Detail the mismatch if any
  if (!match) {
    printf("Difference in index: %d", index_diff);
    printf("\nstd::sort: %d", h_out_cpu[index_diff]);
    printf("\nRadix Sort: %d", out_test[index_diff]);
    int window_sz = 10;
    printf("\nContents: \nstd::sort: ");
    for (int i = -(window_sz / 2); i < (window_sz / 2); ++i) {
      printf("%d, ", h_out_cpu[index_diff + i]);
    }
    printf("\nRadix Sort: ");
    for (int i = -(window_sz / 2); i < (window_sz / 2); ++i) {
      printf("%d, ", out_test[index_diff + i]);
    }
    printf("\n");
  }
  free(h_out_cpu);
}

