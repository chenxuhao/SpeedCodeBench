// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "ctimer.h"
#include "BaseGraph.hh"

void BFSSolver(BaseGraph &g, vidType source, int *dist);
void BFSVerifier(BaseGraph &g, vidType source, int *depth_to_test);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>" << " [source_id(0)] \n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  vidType source = 0;
  if (argc > 2) source = atoi(argv[2]);
  BaseGraph g(argv[1]);
  assert(source >=0 && source < g.V());
  std::vector<int> distances(g.V(), -1);

  ctimer_t t;
  ctimer_start(&t);

  BFSSolver(g, source, &distances[0]);

  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "BFS");

  BFSVerifier(g, source, &distances[0]);

  return 0;
}
