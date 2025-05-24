// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "BaseGraph.hh"

void TCSolver(BaseGraph &g, uint64_t &total);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [oriented(0)]\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/citeseer/graph\n";
    exit(1);
  }
  std::cout << "Triangle Counting: assuming the neighbor lists are sorted.\n";
  int oriented = 1;
  if (argc > 2) oriented = atoi(argv[2]);
  BaseGraph g(argv[1]);
  if (oriented) g.orientation();
  uint64_t total = 0;
  TCSolver(g, total);
  std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}

