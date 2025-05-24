#include "../common/BaseGraph.cc"
#include <vector>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

void TCSolver(BaseGraph &g, uint64_t &total) {
  auto nv = g.V();
  const eidType* _verts = g.rowptr(); // get row pointers array
  const vidType* _edges = g.colidx(); // get column indices array
  std::vector<eidType>verts(_verts, _verts+nv+1);
  std::vector<vidType>edges(_edges, _edges+g.E());
 
  //int num_threads = __cilkrts_get_nworkers();
  //std::cout << "Cilk TC (" << num_threads << " threads)\n";
  cilk::opadd_reducer<uint64_t> counter = 0;
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (vidType u = 0; u < nv; u ++) {
    auto u_adj = &edges[verts[u]];
    auto u_deg = vidType(verts[u+1] - verts[u]);
    for (vidType i = 0; i < u_deg; i++) {
      auto v = u_adj[i];
      auto v_deg = vidType(verts[v+1] - verts[v]);
      auto v_adj = &edges[verts[v]];
      counter += (uint64_t)set_intersect(u_deg, v_deg, u_adj, v_adj);
    }
  }
  total = counter;
  return;
}

