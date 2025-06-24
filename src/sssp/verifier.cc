#include "BaseGraph.hh"
#include <queue>
#include <climits>
typedef int T;
#define kDistInf UINT_MAX/2

void SSSPVerifier(BaseGraph &g, T* weights, vidType source, T *dist_to_test) {
  std::cout << "Verifying SSSP...\n";
  // Serial Dijkstra implementation to get oracle distances
  std::vector<int> oracle_dist(g.V(), kDistInf);
  typedef std::pair<int, vidType> WN;
  std::priority_queue<WN, std::vector<WN>, std::greater<WN> > mq;
  int iter = 0;
  oracle_dist[source] = 0;
  mq.push(std::make_pair(0, source));
  while (!mq.empty()) {
    auto td = mq.top().first;
    auto src = mq.top().second;
    mq.pop();
    if (td == oracle_dist[src]) {
      auto offset = g.edge_begin(src);
      for (auto dst : g.N(src)) {
        auto wt = weights[offset++];
        if (td + wt < oracle_dist[dst]) {
          oracle_dist[dst] = td + wt;
          mq.push(std::make_pair(td + wt, dst));
        }
      }
    }
    iter ++;
  }
  //std::cout << "iterations = " << iter << "\n";
  // Report any mismatches
  bool all_ok = true;
  for (vidType n = 0; n < g.V(); n ++) {
    if (dist_to_test[n] != oracle_dist[n]) {
      std::cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << std::endl;
      all_ok = false;
      break;
    }
  }
  if(all_ok) printf("Correct\n");
  else printf("Wrong\n");
}

