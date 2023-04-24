#include <algorithm>

extern "C"
void cpu_sort(int* out, int* in, size_t len) {
  for (size_t i = 0; i < len; ++i) out[i] = in[i];
  std::sort(out, out + len);
}

