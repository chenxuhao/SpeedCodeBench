#include <omp.h>
#include "BaseGraph.hh"
#include "bitmap.h"
#include "sliding_queue.h"

int64_t BUStep(BaseGraph &g, int *depths, Bitmap &front, Bitmap &next);
int64_t TDStep(BaseGraph &g, int *depths, SlidingQueue<vidType> &queue);

void QueueToBitmap(const SlidingQueue<vidType> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    auto u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(vidType nv, const Bitmap &bm, SlidingQueue<vidType> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<vidType> lqueue(queue);
    #pragma omp for
    for (vidType n = 0; n < nv; n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

void BFSSolver(BaseGraph &g, vidType source, int *depths) {
  auto nv = g.V();

  g.build_reverse_graph();
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  int alpha = 15, beta = 18;
  //std::vector<int> parent(nv);
  //parent[source] = source;
  //std::vector<int> depths(nv);
  #pragma omp parallel for
  for (vidType v = 0; v < nv; v++) {
    int deg = int(g.get_degree(v));
    depths[v] = deg != 0 ? -deg : -1;
  }
  depths[source] = 0;
 
  std::cout << "OpenMP Breadth-first Search (" << num_threads << "threads)\n";
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
        awake_count = BUStep(g, &depths[0], front, curr);
        front.swap(curr);
        printf("BU: iteration=%d, num_frontier=%ld\n", iter, awake_count);
      } while ((awake_count >= old_awake_count) || (awake_count > nv / beta));
      BitmapToQueue(nv, front, queue);
      scout_count = 1;
    } else {
      ++ iter;
      edges_to_check -= scout_count;
      scout_count = TDStep(g, &depths[0], queue);
      queue.slide_window();
      //printf("TD: (scout_count=%ld) ", scout_count);
      printf("TD: iteration=%d, num_frontier=%ld\n", iter, queue.size());
    }
  }

  std::cout << "iterations = " << iter << "\n";
  #pragma omp parallel for
  for (vidType i = 0; i < nv; i ++) {
    if (depths[i] < -1) depths[i] = -1; 
  }
  return;
}

//int64_t BUStep(Graph &g, int *parent, Bitmap &front, Bitmap &next) {
int64_t BUStep(BaseGraph &g, int* depths, Bitmap &front, Bitmap &next) {
  int64_t awake_count = 0;
  next.reset();
  #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  for (vidType dst = 0; dst < g.V(); dst ++) {
    //if (parent[dst] < 0) {
    if (depths[dst] < 0) { // not visited
      for (auto src : g.in_neigh(dst)) {
        if (front.get_bit(src)) {
          //parent[dst] = src;
          depths[dst] = depths[src] + 1;
          awake_count++;
          next.set_bit(dst);
          break;
        }
      }
    }
  }
  return awake_count;
}

//int64_t TDStep(Graph &g, VertexList &parent, SlidingQueue<vidType> &queue) {
int64_t TDStep(BaseGraph &g, int *depths, SlidingQueue<vidType> &queue) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    QueueBuffer<vidType> lqueue(queue);
    #pragma omp for reduction(+ : scout_count)
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      auto src = *q_iter;
      for (auto dst : g.N(src)) {
        //auto curr_val = parent[dst];
        auto curr_val = depths[dst];
        if (curr_val < 0) { // not visited
          //if (compare_and_swap(parent[dst], curr_val, src)) {
          if (compare_and_swap(depths[dst], curr_val, depths[src] + 1)) {
            lqueue.push_back(dst);
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
  }
  return scout_count;
}

