#ifndef PLATFORM_ATOMICS_H_
#define PLATFORM_ATOMICS_H_

#include <atomic>

// gcc/clang/icc instrinsics
template<typename T, typename U>
T fetch_and_add(T &x, U inc) {
  //return __atomic_fetch_add(&x, inc, std::memory_order_seq_cst);
  return __atomic_fetch_add(&x, inc, std::memory_order_relaxed);
}

template<typename T, typename U, typename V>
bool compare_and_swap(T &x, U old_val, V new_val) {
  //return __atomic_compare_exchange_n(&x, &old_val, new_val, false, std::memory_order_seq_cst, std::memory_order_seq_cst);
  return __atomic_compare_exchange_n(&x, &old_val, new_val, false, std::memory_order_relaxed, std::memory_order_relaxed);
}

template <typename ET>
inline bool atomicMin(ET &a, ET b) {
  ET c;
  bool r=0;
  do {
    c = a;
  } while (c > b && !(r=compare_and_swap(a,c,b)));
  return r;
}
#endif  // PLATFORM_ATOMICS_H_
