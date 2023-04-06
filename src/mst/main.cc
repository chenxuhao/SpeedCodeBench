// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void MSTSolver(Graph &g);

int main(int argc, char *argv[]) {
  printf("Minimum Spanning Tree\n");
  if (argc < 2) {
    printf("Usage: %s <graph>\n", argv[0]);
    exit(1);
  }
  Graph g(argv[1], 0, 1, 0, 1); // has edge labels
  g.print_meta_data();
  MSTSolver(g);
  return 0;
}
