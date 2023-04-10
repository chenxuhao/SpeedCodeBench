#include <stdio.h>
#include <algorithm> 
#include <iostream> 
#include <vector> 
#include <chrono>
#include "graph.h"

template <bool weighted, typename T>
void jaccard_weight (GraphF &g, const int iteration);
 
// float or double 
typedef float vtype;

// Forward declarations
template<bool weighted, typename T>
class row_sum;

template<bool weighted, typename T>
class intersection;

template<bool weighted, typename T>
class jw;

template<bool weighted, typename T>
class fill_elements;

// Reference: https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
int main(int argc, char** argv) { 
  int iteration = 10;
  GraphF g(argv[1]);
  g.print_meta_data();
  jaccard_weight<true, vtype>(g, iteration);
  jaccard_weight<false, vtype>(g, iteration);
  return 0; 
} 
