#ifndef C_ATOMICS_H_
#define C_ATOMICS_H_

#include <stdbool.h>
#include <stdint.h>

#define FETCH_ADD(T, name)                            \
  static inline T name(T *x, T inc) {                 \
    return __atomic_fetch_add(x, inc, __ATOMIC_RELAXED); \
  }

#define CAS(T, name)                                               \
  static inline bool name(T *x, T old_val, T new_val) {            \
    return __atomic_compare_exchange_n(x, &old_val, new_val, false, \
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED); \
  }

// Implementations for int, uint8_t, and unsigned char
FETCH_ADD(int, int_fetch_add)
FETCH_ADD(uint8_t, uint8_fetch_add)
FETCH_ADD(unsigned char, uchar_fetch_add)

CAS(int, int_cas)
CAS(uint8_t, uint8_cas)
CAS(unsigned char, uchar_cas)

#endif  // C_ATOMICS_H_
