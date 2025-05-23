// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef SLIDING_QUEUE_H_
#define SLIDING_QUEUE_H_

#include <vector>
#include <algorithm>

#include "platform_atomics.h"

/*
GAP Benchmark Suite
Class:  SlidingQueue
Author: Scott Beamer

Double-buffered queue so appends aren't seen until SlideWindow() called
 - Use QueueBuffer when used in parallel to avoid false sharing by doing
   bulk appends from thread-local storage
*/


template <typename T>
class QueueBuffer;

template <typename T>
class LocalBuffer;

template <typename T>
class SlidingQueue {
  std::vector<T> shared;
  size_t shared_in;
  size_t shared_out_start;
  size_t shared_out_end;
  friend class QueueBuffer<T>;
  friend class LocalBuffer<T>;

 public:
  explicit SlidingQueue(size_t shared_size) {
    shared.resize(shared_size);
    reset();
  }

  ~SlidingQueue() {
  }

  void push_back(T to_add) {
    shared[shared_in++] = to_add;
  }

  bool empty() const {
    return shared_out_start == shared_out_end;
  }

  void reset() {
    shared_out_start = 0;
    shared_out_end = 0;
    shared_in = 0;
  }

  void slide_window() {
    shared_out_start = shared_out_end;
    shared_out_end = shared_in;
  }

  typedef T* iterator;
  typedef const T* const_iterator;

  iterator begin() {
    T* ptr = shared.data();
    return ptr + shared_out_start;
  }

  iterator end() {
    return shared.data() + shared_out_end;
  }

  const_iterator begin() const {
    return shared.data() + shared_out_start;
  }

  const_iterator end() const {
    return shared.data() + shared_out_end;
  }

  size_t size() const {
    return end() - begin();
  }
};


template <typename T>
class QueueBuffer {
  size_t in;
  std::vector<T> local_queue;
  SlidingQueue<T> &sq;
  const size_t local_size;

 public:
  explicit QueueBuffer(SlidingQueue<T> &master, size_t given_size = 16384)
      : sq(master), local_size(given_size) {
    in = 0;
    local_queue.resize(local_size);
  }

  ~QueueBuffer() { }

  void push_back(T to_add) {
    if (in == local_size)
      flush();
    local_queue[in++] = to_add;
  }

  void flush() {
    size_t copy_start = fetch_and_add(sq.shared_in, in);
    std::copy(&local_queue[0], &local_queue[in], &(sq.shared[copy_start]));
    in = 0;
  }
};

template<typename T>
struct VECTOR_PADDED {
  int64_t padding[8];
  std::vector<T> vec;
  int64_t padding2[8];
};

template<typename T>
class LocalBuffer {
  int nthreads;
  const size_t max_size;
  SlidingQueue<T> &sq; // shared queue
  std::vector<VECTOR_PADDED<T>> buffers;
public:
  LocalBuffer(SlidingQueue<T> &q, int nt, size_t local_buf_size = 16384) :
      nthreads(nt), max_size(local_buf_size), sq(q) {
    buffers.resize(nthreads);
    for(int i = 0; i < nthreads; i++) buffers[i].vec.clear();
    reserve(max_size);
  }
  __attribute__((always_inline)) void push_back(int wid, T el) {
    if (buffers[wid].vec.size() == max_size) flush(wid);
    buffers[wid].vec.push_back(el);
  }
  void reserve(int64_t n) {
    for(int i = 0; i < nthreads; i++) {
      buffers[i].vec.reserve(n);
    }
  }
  void flush(int wid) {
    auto in = buffers[wid].vec.size();
    size_t copy_start = fetch_and_add(sq.shared_in, in);
    std::copy(buffers[wid].vec.begin(), buffers[wid].vec.end(), &(sq.shared[0])+copy_start);
    buffers[wid].vec.clear();
  }
  void collect() {
    std::vector<int64_t> offsets(nthreads+1);
    offsets[0] = 0;
    for (int i = 1; i <= nthreads; i++) {
      offsets[i] = offsets[i-1] + buffers[i-1].vec.size();
    }
    for (int i = 0; i < nthreads; i++) {
      int64_t off = offsets[i];
      std::copy(buffers[i].vec.begin(), buffers[i].vec.end(), &(sq.shared[sq.shared_in])+off);
    }
    sq.shared_in += offsets[nthreads];
  }
};

#endif  // SLIDING_QUEUE_H_
