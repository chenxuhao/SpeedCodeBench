#include "BaseGraph.hh"
#include "spmv_util.h"
#include "ctimer.h"
#include <fstream>

typedef float T;
void SpmvSolver(size_t m, size_t nnz, const eidType *Ap, const vidType *Aj, const T *Ax, const T *x, T *y);

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
  srand(13);
  std::string prefix = argv[1];
  std::string Ax_filename = prefix + ".elabel.bin";
  std::ifstream fptr(Ax_filename.c_str());
  if (fptr.good()) {
    T* Ax_ptr = Ax.data();
    read_file(Ax_filename, Ax_ptr, g.E());
  } else {
    for (eidType i = 0; i < g.E(); i++) {
      Ax[i] = rand() / (RAND_MAX + 1.0);
    }
  }
  std::vector<T> x(g.V(), 0);
  std::vector<T> y(g.V(), 0);
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

