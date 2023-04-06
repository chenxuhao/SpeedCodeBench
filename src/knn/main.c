#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

const int N = 10000; // number of data points
const int D = 100;   // number of dimensions
const int K = 10;    // number of neighbors to consider

// Generate random data
void generate_data(float* data) {
  for (int i = 0; i < N * D; i++) {
    data[i] = ((float) rand() / RAND_MAX) * 2 - 1;
  }
}

// Compute Euclidean distance between two data points
float distance(float* a, float* b) {
  float d = 0;
  for (int i = 0; i < D; i++) {
    float diff = a[i] - b[i];
    d += diff * diff;
  }
  return sqrt(d);
}

// Perform k-NN classification
void knn(float* data, int* labels, float* test_point, int *prediction) {
  // Compute distances between test point and all data points
  float distances[N];
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    distances[i] = distance(data + i * D, test_point);
  }

  // Find the K nearest neighbors
  int indices[K];
  for (int i = 0; i < K; i++) {
    float min_distance = INFINITY;
    int min_index = 0;
    for (int j = 0; j < N; j++) {
      if (distances[j] < min_distance) {
        bool already_selected = false;
        for (int k = 0; k < i; k++) {
          if (indices[k] == j) {
            already_selected = true;
            break;
          }
        }
        if (!already_selected) {
          min_distance = distances[j];
          min_index = j;
        }
      }
    }
    indices[i] = min_index;
  }

  // Count the labels of the K nearest neighbors
  int counts[2] = {0, 0};
  for (int i = 0; i < K; i++) {
    counts[labels[indices[i]]]++;
  }

  // Make a prediction based on the majority label
  if (counts[0] > counts[1]) {
    *prediction = 0;
  } else {
    *prediction = 1;
  }
}

int main() {
  float* data = (float*)malloc(N * D * sizeof(float));
  int* labels = (int*)malloc(N* sizeof(int));
  generate_data(data);
  for (int i = 0; i < N; i++) {
    labels[i] = rand() % 2;
  }

  // Perform k-NN classification on a test point
  float test_point[D];
  for (int i = 0; i < D; i++) {
    test_point[i] = ((float) rand() / RAND_MAX) * 2 - 1;
  }
  int prediction;
  knn(data, labels, test_point, &prediction);
  printf("Prediction: %d\n", prediction);
  free(data);
  free(labels);
  return 0;
}

