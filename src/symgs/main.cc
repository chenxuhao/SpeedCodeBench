// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "symgs_util.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
//#include <numeric>

typedef float T;

int ColorSolver(GraphF &g, int *colors);

void SymGSSolver(GraphF &g, vidType *indices, T *x, T *b, std::vector<int> color_offsets);

int main(int argc, char *argv[]) {
  std::cout << "Symmetric Gauss-Seidel smoother by Xuhao Chen\n"
            << "Note: this application uses incoming edge-list, "
            << "which requires reverse (transposed) graph\n";
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph-prefix> \n";
    std::cout << "Example: " << argv[0] << " ../../input/citeseer/graph\n";
    exit(1);
  }
  GraphF g(argv[1], 0, 0, 0, 1, 1, 0, 0);
  g.print_meta_data();

  auto m = g.V();
  auto h_x = custom_alloc_global<T>(m);
  auto h_b = custom_alloc_global<T>(m);
  auto x_host = custom_alloc_global<T>(m);

  // fill matrix with random values: some matrices have extreme values,
  // which makes correctness testing difficult, especially in single precision
  srand(13);
  for(vidType i = 0; i < m; i++) {
    h_x[i] = rand() / (RAND_MAX + 1.0);
    x_host[i] = h_x[i];
  }
  for(vidType i = 0; i < m; i++) 
    h_b[i] = rand() / (RAND_MAX + 1.0);

  //VertexList ordering(m);
  //thrust::sequence(ordering.begin(), ordering.end());
  vidType *ordering = (vidType *)malloc(m * sizeof(vidType));
  #pragma omp parallel for
  for(vidType i = 0; i < m; i++)
    ordering[i] = i;

  // identify parallelism using vertex coloring
  int *colors = (int *)malloc(m * sizeof(int));
  #pragma omp parallel for
  for (vidType i = 0; i < m; i ++)
    colors[i] = MAX_COLOR;
  int num_colors = ColorSolver(g, colors);
  thrust::sort_by_key(colors, colors+m, &ordering[0]);
  int *temp = (int *)malloc((num_colors+1) * sizeof(int));
  thrust::reduce_by_key(colors, colors+m,
                        thrust::constant_iterator<int>(1), 
                        thrust::make_discard_iterator(), temp);
  thrust::exclusive_scan(temp, temp+num_colors+1, temp, 0);
  //std::exclusive_scan(temp, temp+num_colors+1, temp, 0);
  std::vector<int> color_offsets(num_colors+1);
  #pragma omp parallel for
  for (size_t i = 0; i < color_offsets.size(); i ++)
    color_offsets[i] = temp[i];
  SymGSSolver(g, &ordering[0], h_x, h_b, color_offsets);
  SymGSVerifier<T>(g, &ordering[0], h_x, x_host, h_b, color_offsets);
  return 0;
}

