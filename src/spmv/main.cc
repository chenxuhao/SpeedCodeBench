#include "BaseGraph.hh"
#include "spmv_util.h"
#include "ctimer.h"

typedef float T;

extern "C"
void SpmvSolver(size_t m, size_t nnz, 
                const eidType *Ap, const vidType *Aj, 
                const T *Ax, const T *x, T *y);

int main(int argc, char *argv[]) {
  printf("Sparse Matrix-Vector Multiplication\n");
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph-prefix>\n";
    std::cout << "Example: " << argv[0] << " inputs/citeseer\n";
    exit(1);
  }
  BaseGraph g(argv[1]);
  g.build_reverse_graph();
  auto m = g.V();
  auto nnz = g.E();
  std::vector<T> Ax(g.E());
  std::string arr_fname = std::string(argv[1]) + ".elabel.bin";
  load_array(arr_fname, Ax);
  std::vector<T> x(g.V(), 0);
  std::vector<T> y(g.V(), 0);
  srand(13);
  for(vidType i = 0; i < g.V(); i++) {
    x[i] = rand() / (RAND_MAX + 1.0);
    //y[i] = rand() / (RAND_MAX + 1.0);
    //x[i] = 0.3;
  }
  auto Ap = g.in_rowptr();
  auto Aj = g.in_colidx();

  ctimer_t t;
  ctimer_start(&t);

  SpmvSolver(m, nnz, Ap, Aj, &Ax[0], x.data(), y.data());

  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "SpMV");

  double time = (double)timespec_nsec(t.elapsed) / 1e9;
  print_throughput(m, nnz, time);

  SpmvVerifier<T>(m, nnz, Ap, Aj, &Ax[0], x.data(), y.data());
  return 0;
}

