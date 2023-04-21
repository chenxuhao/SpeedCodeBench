// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

void bfs_step(Graph &g, vidType *depth, SlidingQueue<vidType> &queue);

void BFSSolver(Graph &g, vidType source, vidType* dist) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk BFS (" << num_threads << " threads)\n";
  VertexList depth(g.V(), MYINFINITY);
  depth[source] = 0;
  int iter = 0;
  Timer t;
  t.Start();
  SlidingQueue<vidType> queue(g.E());
  queue.push_back(source);
  queue.slide_window();
  while (!queue.empty()) {
    ++ iter;
    std::cout << "iteration=" << iter << ", frontier_size=" << queue.size() << "\n";
    bfs_step(g, depth.data(), queue);
    queue.slide_window();
  }
  t.Stop();
  std::cout << "iterations = " << iter << "\n";
  std::cout << "runtime [cilk_base] = " << t.Seconds() << " sec\n";
}

void bfs_step(Graph &g, vidType *depth, SlidingQueue<vidType> &queue) {
  int num_threads = __cilkrts_get_nworkers();
  //std::vector<QueueBuffer<vidType>> lqueues(num_threads, queue);
  //for (int i = 0; i < num_threads; i ++)
  //  lqueues[i].init(queue);
  auto ptr = queue.begin();
  auto num = queue.size();
  cilk_for (int i = 0; i < num; i++) {
    //int tid = __cilkrts_get_worker_number();
    QueueBuffer<vidType> lqueue(queue); // TODO: overhead of memory allocation
    auto src = ptr[i];
    for (auto dst : g.N(src)) {
      //int curr_val = parent[dst];
      auto curr_val = depth[dst];
      if (curr_val == MYINFINITY) { // not visited
        //if (compare_and_swap(parent[dst], curr_val, src)) {
        if (compare_and_swap(depth[dst], curr_val, depth[src] + 1)) {
          //lqueues[tid].push_back(dst);
          lqueue.push_back(dst);
        }
      }
    }
    //lqueues[tid].flush();
    lqueue.flush();
  }
}

