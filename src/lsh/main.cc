#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

using namespace std;

#define DIMENSION 1000
#define NUM_VECTORS 1000
#define NUM_BUCKETS 100

int hash_function(float *v) {
  int h = 0;
  for (int i = 0; i < DIMENSION; i++) {
    h += v[i] * (i+1);
  }
  return abs(h) % NUM_BUCKETS;
}

int main() {
  float data[NUM_VECTORS][DIMENSION];
  int buckets[NUM_BUCKETS];

  // Generate random data
  srand(time(NULL));
  for (int i = 0; i < NUM_VECTORS; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      data[i][j] = (float) rand() / RAND_MAX;
    }
  }

  // Initialize buckets to zero
  for (int i = 0; i < NUM_BUCKETS; i++) {
    buckets[i] = 0;
  }

  // Compute hash values and increment bucket counts
  #pragma omp parallel for
  for (int i = 0; i < NUM_VECTORS; i++) {
    int hash = hash_function(data[i]);
    #pragma omp atomic
    buckets[hash]++;
  }

  // Print bucket counts
  for (int i = 0; i < NUM_BUCKETS; i++) {
    cout << "Bucket " << i << ": " << buckets[i] << endl;
  }

  return 0;
}

