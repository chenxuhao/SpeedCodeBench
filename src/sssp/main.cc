// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

typedef int ValueT;

void SSSPSolver(Graph &g, vidType source, ValueT *dist, int delta);

void SSSPVerifier(Graph &g, vidType source, int *dist);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>"
      << " [source_id(0)] [delta(-1)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  vidType source = 0;
  if (argc > 2) source = atoi(argv[2]);
  std::cout << "Source vertex: " << source << "\n";
  int delta = 4;
  if (argc > 3) delta = atoi(argv[3]);
  assert(delta > 0);
  std::cout << "Single-source Shortest Paths\n";
  Graph g(argv[1], 0, 1, 0, 1);
  g.print_meta_data();
  assert(source >=0 && source < g.V());
  std::vector<ValueT> distances(g.V(), kDistInf);
  SSSPSolver(g, source, &distances[0], delta);
  SSSPVerifier(g, source, &distances[0]);
  return 0;
}
