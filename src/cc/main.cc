#include "BaseGraph.hh"
#include "ctimer.h"

typedef int comp_t;

void CCSolver(BaseGraph &g, comp_t *comp);
bool CCVerifier(BaseGraph &g, comp_t *comp_test);
int serial_solver(BaseGraph &g, comp_t *components);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> \n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  BaseGraph g(argv[1]);
  g.build_reverse_graph();
  std::vector<comp_t> comp(g.V());
  // Initialize each node to a single-node self-pointing tree
  for (vidType i = 0; i < g.V(); i++) comp[i] = i;

  ctimer_t t;
  ctimer_start(&t);
  CCSolver(g, &comp[0]);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "Connected Components");

  bool correct = CCVerifier(g, &comp[0]);
  if (correct) printf("Correct\n");
  else printf("Wrong\n");

  std::vector<comp_t> s_comp(g.V(), -1);
  ctimer_start(&t);
  serial_solver(g, s_comp.data());
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "Serial Connected Components");

  return 0;
}
