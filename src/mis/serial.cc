// Copyright 2020
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void MISSolver(Graph &g, VertexList &mis) {
  std::cout << "Serial MIS\n";
  Timer t;
  t.Start();
  // add your code here
  
  t.Stop();
  std::cout << "runtime [serial] = " << t.Seconds() << " sec\n";
}

