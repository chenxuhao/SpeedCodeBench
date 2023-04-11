#include "graph.h"
#include "spmv_util.h"

typedef float T;
void SpmvSolver(GraphF &g, const T *x, T *y);

int main(int argc, char *argv[]) {
  printf("Sparse Matrix-Vector Multiplication\n");
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph-prefix>\n";
    std::cout << "Example: " << argv[0] << " inputs/citeseer\n";
    exit(1);
  }
  GraphF g(argv[1], 0, 0, 0, 1, 1, 0, 0);
  g.print_meta_data();
  std::vector<T> x(g.V(), 0);
  std::vector<T> y(g.V(), 0);
  //srand(13);
  for(vidType i = 0; i < g.V(); i++) {
    //x[i] = rand() / (RAND_MAX + 1.0);
    //y[i] = rand() / (RAND_MAX + 1.0);
    x[i] = 0.3;
  }

  SpmvSolver(g, x.data(), y.data());
  SpmvVerifier<T>(g, x.data(), y.data());
  return 0;
}

