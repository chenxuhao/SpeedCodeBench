// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void MISSolver(Graph &g, VertexList &mis);
void MISVerifier(Graph &g, VertexList &mis);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [oriented(0)]\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/citeseer/graph\n";
    exit(1);
  }
  std::cout << "Maximal Independent Set (MIS)\n";
  Graph g(argv[1]);
  g.print_meta_data();
  VertexList mis;
  MISSolver(g, mis);
  MISVerifier(g, mis);
  return 0;
}

