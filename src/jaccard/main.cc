#include <stdio.h>
#include <algorithm> 
#include <iostream> 
#include <vector> 
#include <chrono>
#include "graph.h"

template <bool weighted, typename T>
void jaccard_weight (GraphF &g, const int iteration);

// Reference: https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
int main(int argc, char** argv) { 
  int iteration = 10;
  GraphF g(argv[1], 0, 0, 0, 1, 1, 0, 0);
  g.print_meta_data();
  jaccard_weight<true, float>(g, iteration);
  jaccard_weight<false, float>(g, iteration);
  return 0; 
}
