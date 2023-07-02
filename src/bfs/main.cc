// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void BFSSolver(Graph &g, vidType source, int *dist);
void BFSVerifier(Graph &g, vidType source, int *depth_to_test);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>"
      << " [source_id(0)] [reverse(0)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  vidType source = 0;
  int reverse = 0;
  if (argc > 2) source = atoi(argv[2]);
  if (argc > 3) reverse = atoi(argv[3]);
  Graph g(argv[1], 0, 1, 0, 0, reverse);
  g.print_meta_data();
  assert(source >=0 && source < g.V());
  std::vector<int> distances(g.V(), -1);
  BFSSolver(g, source, &distances[0]);
  BFSVerifier(g, source, &distances[0]);
  return 0;
}
