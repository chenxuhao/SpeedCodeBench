// Copyright 2023, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <cilk/cilk.h>
//#include <random>
#include "platform_atomics.h"

int SampleFrequentElement(vidType m, int *comp, int64_t num_samples = 1024);
void Link(vidType u, vidType v, int *comp);
void Compress(vidType m, int *comp);

void CCSolver(Graph &g, int *comp) {
  if (!g.has_reverse_graph()) {
    std::cout << "The Afforest algorithm requires the reverse graph (incoming edges)\n";
    std::cout << "Please set reverse to 1 in the command line\n";
    exit(1);
  }
  cilk_for (vidType i = 0; i < g.V(); i++) comp[i] = i;
  const vidType m = g.V();
  int neighbor_rounds = 2;

  Timer t;
  t.Start();
  // Process a sparse sampled subgraph first for approximating components.
  // Sample by processing a fixed number of neighbors for each vertex
  for (int r = 0; r < neighbor_rounds; ++r) {
    cilk_for (vidType src = 0; src < m; src ++) {
      for (vidType dst : g.out_neigh(src, r)) {
        // Link at most one time if neighbor available at offset r
        Link(src, dst, comp);
        break;
      }
    }
    Compress(m, comp);
  }

  // Sample 'comp' to find the most frequent element -- due to prior
  // compression, this value represents the largest intermediate component
  auto c = SampleFrequentElement(m, comp);

  // Final 'link' phase over remaining edges (excluding largest component)
  if (!g.is_directed()) {
    cilk_for (vidType u = 0; u < m; u ++) {
      // Skip processing nodes in the largest component
      if (comp[u] == c) continue;
      // Skip over part of neighborhood (determined by neighbor_rounds)
      for (auto v : g.out_neigh(u, neighbor_rounds)) {
        Link(u, v, comp);
      }
    }
  } else {
    cilk_for (vidType u = 0; u < m; u ++) {
      if (comp[u] == c) continue;
      for (auto v : g.out_neigh(u, neighbor_rounds)) {
        Link(u, v, comp);
      }
      // To support directed graphs, process reverse graph completely
      for (auto v : g.in_neigh(u)) {
        Link(u, v, comp);
      }
    }
  }
  // Finally, 'compress' for final convergence
  Compress(m, comp);
  t.Stop();
  std::cout << "runtime [cilk_afforest] = " << t.Seconds() << " seconds\n";
}

// Place nodes u and v in same component of lower component ID
void Link(vidType u, vidType v, int *comp) {
  auto p1 = comp[u];
  auto p2 = comp[v];
  while (p1 != p2) {
    auto high = p1 > p2 ? p1 : p2;
    auto low = p1 + (p2 - high);
    auto p_high = comp[high];
    // Was already 'low' or succeeded in writing 'low'
    if ((p_high == low) || (p_high == high && compare_and_swap(comp[high], high, low)))
      break;
    p1 = comp[comp[high]];
    p2 = comp[low];
  }
}

// Reduce depth of tree for each component to 1 by crawling up parents
void Compress(vidType m, int *comp) {
  cilk_for (vidType n = 0; n < m; n++) {
    while (comp[n] != comp[comp[n]]) {
      comp[n] = comp[comp[n]];
    }
  }
}

