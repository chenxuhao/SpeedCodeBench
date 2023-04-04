// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <queue>

void SSSPVerifier(Graph &g, vidType source, elabel_t *dist_to_test) {
  std::cout << "Verifying SSSP...\n";
  // Serial Dijkstra implementation to get oracle distances
  vector<elabel_t> oracle_dist(g.V(), kDistInf);
  typedef pair<elabel_t, IndexT> WN;
  std::priority_queue<WN, vector<WN>, greater<WN> > mq;
  int iter = 0;
  Timer t;
  t.Start();
  oracle_dist[source] = 0;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    elabel_t td = mq.top().first;
    IndexT src = mq.top().second;
    mq.pop();
    if (td == oracle_dist[src]) {
      auto offset = g.edge_begin(src);
      for (auto dst : g.N(src)) {
        elabel_t wt = g.getEdgeData(offset++);
        if (td + wt < oracle_dist[dst]) {
          oracle_dist[dst] = td + wt;
          mq.push(make_pair(td + wt, dst));
        }
      }
    }
    iter ++;
  }
  t.Stop();
  //std::cout << "iterations = " << iter << "\n";
  std::cout << "runtime [serial] = " << t.Seconds() << " sec\n";

  // Report any mismatches
  bool all_ok = true;
  for (vidType n = 0; n < g.V(); n ++) {
    if (dist_to_test[n] != oracle_dist[n]) {
      //std::cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << std::endl;
      all_ok = false;
    }
  }
  if(all_ok) printf("Correct\n");
  else printf("Wrong\n");
}

