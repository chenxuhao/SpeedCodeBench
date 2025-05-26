#ifndef _OPMAX_REDUCER_H
#define _OPMAX_REDUCER_H

#ifdef __cplusplus
#include <limits>
#include <algorithm>

namespace cilk {

template <typename T> static void max_init(void *v) {
    *static_cast<T *>(v) = static_cast<T>(std::numeric_limits<T>::min());
}

template <typename T> static void max_reduce(void *l, void *r) {
    *static_cast<T *>(l) = std::max(*static_cast<T *>(l), *static_cast<T *>(r));
}

template <typename T> using opmax_reducer = T _Hyperobject(max_init<T>, max_reduce<T>);

} // namespace cilk

#endif // #ifdef __cplusplus

#endif // _OPMAX_REDUCER_H
