#include <iostream>
#include <algorithm>

void cpu_sort(unsigned int* out, unsigned int* in, size_t len) {
  for (size_t i = 0; i < len; ++i) out[i] = in[i];
  std::sort(out, out + len);
}

extern "C"
void verify(unsigned int* out, unsigned int* const in, unsigned int num_elems) {
  // Check for any mismatches between outputs of radix-sort and std::sort
  unsigned int* h_out_cpu = new unsigned int[num_elems];
  cpu_sort(h_out_cpu, in, num_elems);
  bool match = true;
  int index_diff = 0;
  for (unsigned int i = 0; i < num_elems; ++i) {
    if (h_out_cpu[i] != out[i]) {
      match = false;
      index_diff = i;
      break;
    }
  }
  if (match) std::cout << "PASS\n";
  else std::cout << "Mismatch!\n";

  // Detail the mismatch if any
  if (!match) {
    std::cout << "Difference in index: " << index_diff << std::endl;
    std::cout << "std::sort: " << h_out_cpu[index_diff] << std::endl;
    std::cout << "Radix Sort: " << out[index_diff] << std::endl;
    int window_sz = 10;
    std::cout << "Contents: " << std::endl;
    std::cout << "std::sort: ";
    for (int i = -(window_sz / 2); i < (window_sz / 2); ++i) {
      std::cout << h_out_cpu[index_diff + i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Radix Sort: ";
    for (int i = -(window_sz / 2); i < (window_sz / 2); ++i) {
      std::cout << out[index_diff + i] << ", ";
    }
    std::cout << std::endl;
  }
  delete[] h_out_cpu;
}

