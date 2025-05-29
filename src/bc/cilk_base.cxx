#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "BaseGraph.hh"
#include "bitmap.h"
#include "opmax_reducer.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

typedef float score_t; // vertex label

void PBFS(BaseGraph &g, int source, std::vector<int> &path_counts, 
          std::vector<int> &depths, Bitmap &succ,
          std::vector<SlidingQueue<vidType>::iterator> &depth_index, 
          SlidingQueue<vidType> &queue) {
  auto nthreads = __cilkrts_get_nworkers();
  std::cout << "Cilk BFS (" << nthreads << " threads)\n";

  depths[source] = 0;
  path_counts[source] = 1;
  queue.push_back(source);
  depth_index.push_back(queue.begin());
  queue.slide_window();
  int depth = 0;
  while (!queue.empty()) {
    ++ depth;
    depth_index.push_back(queue.begin());
    LocalBuffer<vidType> lqueues(queue, nthreads);
    vidType* ptr = queue.begin();
    printf("Forward: depth=%d, frontire_size=%d\n", depth, queue.size());
    cilk_for (int i = 0; i < queue.size(); i++) {
      auto tid = __cilkrts_get_worker_number();
      auto u = ptr[i];
      auto offset = g.edge_begin(u);
      for (auto v : g.N(u)) {
        if (depths[v] == -1  && compare_and_swap(depths[v], -1, depths[u] + 1)) {
          lqueues.push_back(tid, v);
        }
        if (depths[v] == depth) {
          succ.set_bit_atomic(offset);
          fetch_and_add(path_counts[v], path_counts[u]);
        }
        offset ++;
      }
    }
    lqueues.collect();
    queue.slide_window();
  }
  depth_index.push_back(queue.begin());
}

void BCSolver(BaseGraph &g, vidType source, score_t *scores) {
  auto m = g.V();
  int num_iters = 1;
  Bitmap succ(g.E());
  std::vector<SlidingQueue<vidType>::iterator> depth_index;

  int depth = 0;
  SlidingQueue<vidType> queue(m);
  for (int iter = 0; iter < num_iters; iter++) {
    std::vector<int> path_counts(m, 0);
    std::vector<int> depths(m, -1);
    depth_index.resize(0);
    queue.reset();
    succ.reset();
    PBFS(g, source, path_counts, depths, succ, depth_index, queue);
    std::vector<score_t> deltas(m, 0);
    for (int d = depth_index.size()-2; d >= 0; d --) {
      depth ++;
      auto nitems = depth_index[d+1] - depth_index[d];
      printf("Reverse: depth=%d, frontier_size=%ld\n", d, nitems);
      cilk_for (vidType *it = depth_index[d]; it < depth_index[d+1]; it++) {
        auto src = *it;
        score_t delta_src = 0;
        auto offset = g.edge_begin(src);
        for (auto dst : g.N(src)) {
          if (succ.get_bit(offset)) {
            delta_src += static_cast<score_t>(path_counts[src]) /
              static_cast<score_t>(path_counts[dst]) * (1 + deltas[dst]);
          }
          offset ++;
        }
        deltas[src] = delta_src;
        scores[src] += delta_src;
      }
    }
  }
  // Normalize scores
  score_t biggest_score = 0;
  cilk::opmax_reducer<score_t> max_score = 0;
  cilk_for (vidType n = 0; n < m; n ++)
    max_score = std::max<score_t>(max_score, scores[n]);
  biggest_score = max_score;
  cilk_for (vidType n = 0; n < m; n ++)
    scores[n] = scores[n] / biggest_score;
  std::cout << "iterations = " << depth << ".\n";
}
