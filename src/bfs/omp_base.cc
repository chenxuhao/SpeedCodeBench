#include <omp.h>
#include "BaseGraph.hh"
#include "bitmap.h"
#include "sliding_queue.h"

void bfs_step(BaseGraph &g, int *depth, SlidingQueue<vidType> &queue) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  LocalBuffer<vidType> lqueue(queue, num_threads);
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    auto tid = omp_get_thread_num();
    auto src = *q_iter;
    for (auto dst : g.N(src)) {
      auto curr_val = depth[dst];
      if (curr_val == -1) { // not visited
        if (compare_and_swap(depth[dst], curr_val, depth[src] + 1)) {
          lqueue.push_back(tid, dst);
        }
      }
    }
  }
  lqueue.collect();
}
/*
void bfs_step(Graph &g, int *depth, SlidingQueue<vidType> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<vidType> lqueue(queue);
    #pragma omp for
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      auto src = *q_iter;
      for (auto dst : g.N(src)) {
        //int curr_val = parent[dst];
        auto curr_val = depth[dst];
        if (curr_val == -1) { // not visited
          //if (compare_and_swap(parent[dst], curr_val, src)) {
          if (compare_and_swap(depth[dst], curr_val, depth[src] + 1)) {
            lqueue.push_back(dst);
          }
        }
      }
    }
    lqueue.flush();
  }
}
*/
void BFSSolver(BaseGraph &g, vidType source, int* depth) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP BFS (" << num_threads << " threads)\n";
  depth[source] = 0;
  int iter = 0;
  SlidingQueue<vidType> queue(g.E());
  queue.push_back(source);
  queue.slide_window();
  while (!queue.empty()) {
    ++ iter;
    std::cout << "iteration=" << iter << ", frontier_size=" << queue.size() << "\n";
    bfs_step(g, depth, queue);
    queue.slide_window();
  }
  std::cout << "iterations = " << iter << "\n";
}

