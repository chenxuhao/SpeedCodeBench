#include "../common/BaseGraph.cc"
#include "sliding_queue.h"
#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>
#include "ctimer.h"
#include <memory>

void BFSSolver(BaseGraph &g, vidType source, int *_depths) {
  //auto nthreads = __cilkrts_get_nworkers();
  //std::cout << "Cilk BFS (" << nthreads << " threads)\n";
  _depths[source] = 0;

  auto nv = g.V();
  const eidType* _verts = g.rowptr(); // get row pointers array
  const vidType* _edges = g.colidx(); // get column indices array
  std::vector<eidType>verts(_verts, _verts+nv+1);
  std::vector<vidType>edges(_edges, _edges+g.E());
  std::vector<int>depths(_depths, _depths+nv);

  ctimer_t t;
  ctimer_start(&t);

  std::vector<vidType> frontier(nv);
  std::vector<vidType> next_frontier(nv);
  frontier[0] = source;
  size_t frontier_size = 1;
 
  int iter = 0;
  while (frontier_size > 0) {
    ++ iter;
    std::cout << "iteration=" << iter << ", frontier_size=" << frontier_size << "\n";
    //size_t index = 0;
    std::unique_ptr<size_t> index = std::make_unique<size_t>(0);
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (int i = 0; i < frontier_size; i++) {
      auto u = frontier[i];
      auto u_adj = &edges[verts[u]];
      auto u_deg = vidType(verts[u+1] - verts[u]);
      //printf("u=%d\n", u);
      for (vidType j = 0; j < u_deg; j++) {
        auto v = u_adj[j];
        if (depths[v] == -1) {
          // make sure each unique neighbor is inserted to the frontier only once
          if (compare_and_swap(depths[v], -1, depths[u] + 1)) {
            // avoid data race on frontier insertion
            size_t pos = fetch_and_add(*index, 1);
            //assert(pos < nv);
            next_frontier[pos%nv] = v;
          }
        }
      }
    }
    frontier_size = *index;
    std::swap(frontier, next_frontier);
  }

  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "BFS-zera_base-kernel");

  std::cout << "iterations = " << iter << "\n";
  std::copy(depths.begin(), depths.end(), _depths);
}

