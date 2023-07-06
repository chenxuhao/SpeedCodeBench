// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "pr.h"

void PRSolver(Graph &g, score_t *scores);
void PRVerifier(Graph &g, score_t *scores, double target_error);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)]"
      << " [symmetrize(0/1)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  int reverse = 1;
  if (argc > 2) reverse = atoi(argv[2]);
  std::cout << "PageRank: taking directed graphs only, undirected graphs are treated as bidirected\n";
  Graph g(argv[1], 0 , 1, 0, 0, reverse);
  g.print_meta_data();

  const score_t init_score = 1.0f / g.V();
  std::cout << "PageRank: initial score = " << init_score << "\n";
  std::vector<score_t> scores(g.V(), init_score);
  PRSolver(g, &scores[0]);
  PRVerifier(g, &scores[0], EPSILON);
  return 0;
}

