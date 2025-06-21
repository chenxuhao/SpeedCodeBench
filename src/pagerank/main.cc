#include "pr.h"
#include "ctimer.h"
#include "BaseGraph.hh"

void PRSolver(BaseGraph &g, score_t *scores);
void PRVerifier(BaseGraph &g, score_t *scores, double target_error);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/citeseer/graph\n";
    exit(1);
  }
  std::cout << "PageRank: assuming symmetric (bidirected) graph for simplicity\n";
  BaseGraph g(argv[1]);
  g.build_reverse_graph();

  const score_t init_score = 1.0f / g.V();
  std::cout << "PageRank: initial score = " << init_score << "\n";
  std::vector<score_t> scores(g.V(), init_score);

  ctimer_t t;
  ctimer_start(&t);
  PRSolver(g, &scores[0]);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "PageRank");

  double elap_time = (double)timespec_nsec(t.elapsed) / 1e9;
  std::cout << "throughput = " << double(g.E()) / elap_time / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";

  ctimer_start(&t);
  PRVerifier(g, &scores[0], EPSILON);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "Verify");

  return 0;
}

