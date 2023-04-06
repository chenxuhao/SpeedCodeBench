// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void BCSolver(Graph &g, vidType source, score_t *scores);
void BCVerifier(Graph &g, int source, int num_iters, score_t *scores_to_test);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> " << "[source_id(0)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/citeseer/graph\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  int source = 0;
  if (argc > 2) source = atoi(argv[2]);
  std::cout << "Betweenness Centrality: source vid = " << source << "\n";

  std::vector<score_t> scores(g.V(), 0);
  BCSolver(g, source, &scores[0]);
  BCVerifier(g, source, 1, &scores[0]);
  return 0;
}
