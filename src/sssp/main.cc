#include "BaseGraph.hh"
#include "ctimer.h"
#include <climits>

typedef int T;
#define kDistInf UINT_MAX/2
void SSSPSolver(BaseGraph &g, T *weights, vidType source, T *dist);
void SSSPVerifier(BaseGraph &g, T *weights, vidType source, T *dist);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>"
      << " [source_id(0)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  vidType source = 0;
  if (argc > 2) source = atoi(argv[2]);
  BaseGraph g(argv[1]);
  std::vector<T> edge_weights(g.E());
  std::string arr_fname = std::string(argv[1]) + ".elabel.bin";
  load_array(arr_fname, edge_weights);
  assert(source >=0 && source < g.V());
  std::cout << "Source vertex: " << source << "\n";
  std::vector<T> distances(g.V(), kDistInf);

  ctimer_t t;
  ctimer_start(&t);
  SSSPSolver(g, edge_weights.data(), source, &distances[0]);

  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "SSSP");

  double time = (double)timespec_nsec(t.elapsed) / 1e9;
  
  SSSPVerifier(g, edge_weights.data(), source, &distances[0]);
  return 0;
}
