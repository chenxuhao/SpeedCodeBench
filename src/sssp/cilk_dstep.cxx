// Copyright 2023 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "platform_atomics.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

//[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
//    algorithm." Journal of Algorithms, 49(1):114--152, 2003.
void SSSPSolver(Graph &g, vidType source, int *dist) {
  const int delta = 4;
  dist[source] = 0;
  VertexList frontier(g.E());
  frontier[0] = source;
  auto num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk SSSP (" << num_threads << " threads)\n";
 
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  size_t shared_indexes[2] = {0, kDistInf};
  size_t frontier_tails[2] = {1, 0}; 
  std::vector<VertexLists> per_thread_bins(num_threads);

  Timer t;
  t.Start();
  int iter = 0;
  while (static_cast<int>(shared_indexes[iter&1]) != kDistInf) {
    size_t &curr_bin_index = shared_indexes[iter&1];
    size_t &next_bin_index = shared_indexes[(iter+1)&1];
    size_t &curr_frontier_tail = frontier_tails[iter&1];
    size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
    //std::cout << "frontier size: " << curr_frontier_tail << "\n";
    #pragma cilk grainsize 64
    cilk_for (size_t i = 0; i < curr_frontier_tail; i ++) {
      auto tid = __cilkrts_get_worker_number();
      //auto tid = omp_get_thread_num();
      auto &local_bins = per_thread_bins[tid];
      auto src = frontier[i];
      if (dist[src] >= delta * static_cast<int>(curr_bin_index)) {
        auto offset = g.edge_begin(src);
        for (auto dst : g.N(src)) {
          auto old_dist = dist[dst];
          auto new_dist = dist[src] + g.getEdgeData(offset++);
          if (new_dist < old_dist) {
            bool changed_dist = true;
            while (!compare_and_swap(dist[dst], old_dist, new_dist)) {
              old_dist = dist[dst];
              if (old_dist <= new_dist) {
                changed_dist = false;
                break;
              }
            }
            if (changed_dist) {
              size_t dest_bin = new_dist/delta;
              if (dest_bin >= local_bins.size()) {
                local_bins.resize(dest_bin+1);
              }
              local_bins[dest_bin].push_back(dst);
            }
          }
        }
      }
    }

    // find the next bin with minimum priority
    int max_nbins = 1;
    for (int tid = 0; tid < num_threads; tid++) {
      if (per_thread_bins[tid].size() > max_nbins)
        max_nbins = per_thread_bins[tid].size();
    }
    bool found = false;
    for (size_t i = curr_bin_index; i < max_nbins; i ++) {
      for (int tid = 0; tid < num_threads; tid++) {
        auto &local_bins = per_thread_bins[tid];
        if (i < local_bins.size() && !local_bins[i].empty()) {
          next_bin_index = i;
          found = true;
          break;
        }
      }
      if (found) break;
    }
    //std::cout << "next_bin_index: " << next_bin_index << "\n";

    // re-initialize
    curr_bin_index = kDistInf;
    curr_frontier_tail = 0;

    // dump the local bins (with minimum priority) into the global frontier
    #pragma cilk grainsize 1
    cilk_for (int tid = 0; tid < num_threads; tid++) {
      auto &local_bins = per_thread_bins[tid];
      if (next_bin_index < local_bins.size()) {
        auto &next_bin = local_bins[next_bin_index];
        size_t copy_start = fetch_and_add(next_frontier_tail, next_bin.size());
        copy(next_bin.begin(), next_bin.end(), &frontier[copy_start]);
        next_bin.resize(0);
      }
    }
    iter++;
  }

  t.Stop();
  std::cout << "iterations = " << iter << "\n";
  std::cout << "runtime [cilk_dstep] = " << t.Seconds() << "sec\n";
  return;
}

