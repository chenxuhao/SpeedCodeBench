#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>
#include "ctimer.h"
#include "BaseGraph.hh"
#include "../common/BaseGraph.cc"
#include <memory>

void CCSolver(BaseGraph &g, int *_comp) {
  //int num_threads = __cilkrts_get_nworkers();
  //std::cout << "Cilk Connected Components (" << num_threads << " threads)\n";

  auto nv = g.V();
  const eidType* _verts = g.rowptr(); // get row pointers array
  const vidType* _edges = g.colidx(); // get column indices array
  std::vector<eidType>verts(_verts, _verts+nv+1);
  std::vector<vidType>edges(_edges, _edges+g.E());
  std::vector<int>comp(_comp, _comp+nv);

  ctimer_t t;
  ctimer_start(&t);

  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (vidType n = 0; n < nv; n ++) comp[n] = n;
  int iter = 0;
  //bool change = true;
  std::unique_ptr<bool> change = std::make_unique<bool>(1);
  while (*change) {
    *change = false;
    iter++;
    //printf("Executing iteration %d ...\n", iter);
    //#pragma cilk grainsize 64
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (vidType src = 0; src < nv; src ++) {
      auto comp_src = comp[src];
      //for (auto dst : g.N(src)) {
      auto adj = &edges[verts[src]];
      auto deg = vidType(verts[src+1] - verts[src]);
      for (vidType j = 0; j < deg; j++) {
        auto dst = adj[j];
        auto comp_dst = comp[dst];
        if (comp_src == comp_dst) continue;
        // Hooking condition so lower component ID wins independent of direction
        int high_comp = comp_src > comp_dst ? comp_src : comp_dst;
        int low_comp = comp_src + (comp_dst - high_comp);
        if (high_comp == comp[high_comp]) {
          if (!*change) *change = true;
          comp[high_comp] = low_comp;
        }
      }
    }
    //#pragma cilk grainsize 64
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (vidType n = 0; n < nv; n++) {
      while (comp[n] != comp[comp[n]]) {
        comp[n] = comp[comp[n]];
      }
    }
  }
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "CC-zera_base-kernel");

  std::cout << "iterations = " << iter << "\n";
  std::copy(comp.begin(), comp.end(), _comp);
}
