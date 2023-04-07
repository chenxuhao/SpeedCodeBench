#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void merge(int arr[], int left, int middle, int right) {
  int i, j, k;
  int n1 = middle - left + 1;
  int n2 = right - middle;
  int L[n1], R[n2];

  for (i = 0; i < n1; i++)
    L[i] = arr[left + i];
  for (j = 0; j < n2; j++)
    R[j] = arr[middle + 1 + j];

  i = 0;
  j = 0;
  k = left;

  while (i < n1 && j < n2) {
    if (L[i] <= R[j]) {
      arr[k] = L[i];
      i++;
    }
    else {
      arr[k] = R[j];
      j++;
    }
    k++;
  }

  while (i < n1) {
    arr[k] = L[i];
    i++;
    k++;
  }

  while (j < n2) {
    arr[k] = R[j];
    j++;
    k++;
  }
}

void mergeSort(int arr[], int left, int right) {
  if (left < right) {
    int middle = left + (right - left) / 2;

    #pragma omp parallel sections
    {
      #pragma omp section
      mergeSort(arr, left, middle);
      #pragma omp section
      mergeSort(arr, middle + 1, right);
    }

    merge(arr, left, middle, right);
  }
}

int main() {
  int arr[] = {38, 27, 43, 3, 9, 82, 10};
  int n = sizeof(arr) / sizeof(arr[0]);

  printf("Original array: ");
  for (int i = 0; i < n; i++)
    printf("%d ", arr[i]);

  mergeSort(arr, 0, n - 1);

  printf("\nSorted array: ");
  for (int i = 0; i < n; i++)
    printf("%d ", arr[i]);

  return 0;
}

