// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "BaseGraph.hh"
#include "bitmap.h"
#include "sliding_queue.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

typedef int outType;

int64_t BUStep(BaseGraph &g, outType *depths, Bitmap &front, Bitmap &next);
int64_t TDStep(BaseGraph &g, outType *depths, SlidingQueue<vidType> &queue);
void QueueToBitmap(const SlidingQueue<vidType> &queue, Bitmap &bm);
void BitmapToQueue(vidType nv, const Bitmap &bm, SlidingQueue<vidType> &queue);

void BFSSolver(BaseGraph &g, vidType source, outType *depths) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk BFS using direction (" << num_threads << " threads)\n";
  g.build_reverse_graph();
  auto nv = g.V();
  int alpha = 15, beta = 18;

  cilk_for (vidType v = 0; v < nv; v++) {
    int deg = int(g.get_degree(v));
    depths[v] = (deg != 0) ? -deg : -1;
  }
  depths[source] = 0;
  SlidingQueue<vidType> queue(nv);
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(nv);
  curr.reset();
  Bitmap front(nv);
  front.reset();
  int64_t edges_to_check = g.E();
  int64_t scout_count = g.get_degree(source);
  int iter = 0;
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      QueueToBitmap(queue, front);
      awake_count = queue.size();
      queue.slide_window();
      do {
        ++ iter;
        old_awake_count = awake_count;
        awake_count = BUStep(g, depths, front, curr);
        front.swap(curr);
        printf("BU: iteration=%d, num_frontier=%ld\n", iter, awake_count);
      } while ((awake_count >= old_awake_count) || (awake_count > nv / beta));
      BitmapToQueue(nv, front, queue);
      scout_count = 1;
    } else {
      ++ iter;
      edges_to_check -= scout_count;
      scout_count = TDStep(g, depths, queue);
      queue.slide_window();
      //printf("TD: (scout_count=%ld) ", scout_count);
      printf("TD: iteration=%d, num_frontier=%ld\n", iter, queue.size());
    }
  }
  cilk_for (vidType i = 0; i < nv; i ++)
    if (depths[i] < -1) depths[i] = -1; 
  std::cout << "iterations = " << iter << "\n";
}

int64_t BUStep(BaseGraph &g, outType* depths, Bitmap &front, Bitmap &next) {
  int64_t awake_count = 0;
  cilk::opadd_reducer<int64_t> counter = 0;
  next.reset();
  #pragma cilk grainsize 1024
  cilk_for (vidType u = 0; u < g.V(); u ++) {
    if (depths[u] < 0) { // not visited
      for (auto v : g.in_neigh(u)) {
        if (front.get_bit(v)) {
          assert(depths[v]+1 > 0);
          depths[u] = depths[v] + 1;
          counter ++;
          next.set_bit(u);
          break;
        }
      }
    }
  }
  awake_count = counter;
  return awake_count;
}

int64_t TDStep(BaseGraph &g, outType *depths, SlidingQueue<vidType> &queue) {
  int64_t scout_count = 0;
  cilk::opadd_reducer<int64_t> counter = 0;
  auto nthreads = __cilkrts_get_nworkers();
  LocalBuffer<vidType> lqueues(queue, nthreads);
  vidType* ptr = queue.begin();
  //#pragma cilk grainsize 64
  cilk_for (int i = 0; i < queue.size(); i++) {
    auto tid = __cilkrts_get_worker_number();
    auto u = ptr[i];
    assert(depths[u]+1 > 0);
    for (auto v : g.N(u)) {
      int curr_val = depths[v];
      if (curr_val < 0) {
        if (compare_and_swap(depths[v], curr_val, depths[u] + 1)) {
          lqueues.push_back(tid, v);
          counter += -curr_val;
        }
      }
    }
  }
  lqueues.collect();
  scout_count = counter;
  return scout_count;
}

void QueueToBitmap(const SlidingQueue<vidType> &queue, Bitmap &bm) {
  const vidType* ptr = queue.begin();
  cilk_for (int i = 0; i < queue.size(); i++) {
    bm.set_bit_atomic(ptr[i]);
  }
}

void BitmapToQueue(vidType nv, const Bitmap &bm, SlidingQueue<vidType> &queue) {
  auto nthreads = __cilkrts_get_nworkers();
  LocalBuffer<vidType> lqueues(queue, nthreads); // per-thread local queue
  cilk_for (vidType n = 0; n < nv; n++) {
    auto tid = __cilkrts_get_worker_number();
    if (bm.get_bit(n)) lqueues.push_back(tid, n);
  }
  lqueues.collect();
  queue.slide_window();
}

