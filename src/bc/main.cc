#include "BaseGraph.hh"
#include "ctimer.h"

typedef float score_t; // vertex label

void BCSolver(BaseGraph &g, vidType source, score_t *scores);
void BCVerifier(BaseGraph &g, int source, int num_iters, score_t *scores_to_test);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> " << "[source_id(0)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/citeseer/graph\n";
    exit(1);
  }
  BaseGraph g(argv[1]);
  int source = 0;
  if (argc > 2) source = atoi(argv[2]);
  std::cout << "Betweenness Centrality: source vid = " << source << "\n";

  std::vector<score_t> scores(g.V(), 0);

  ctimer_t t;
  ctimer_start(&t);
  BCSolver(g, source, &scores[0]);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "Betweenness Centrality");

  BCVerifier(g, source, 1, &scores[0]);
  return 0;
}
