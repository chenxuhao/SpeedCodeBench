#ifndef _OPMIN_REDUCER_H
#define _OPMIN_REDUCER_H

#ifdef __cplusplus
#include <limits>
#include <algorithm>

namespace cilk {

template <typename T> static void min_init(void *v) {
    *static_cast<T *>(v) = static_cast<T>(std::numeric_limits<T>::max());
}

template <typename T> static void min_reduce(void *l, void *r) {
    *static_cast<T *>(l) = std::min(*static_cast<T *>(l), *static_cast<T *>(r));
}

template <typename T> using opmin_reducer = T _Hyperobject(min_init<T>, min_reduce<T>);

} // namespace cilk

#endif // #ifdef __cplusplus

#endif // _OPMIN_REDUCER_H
