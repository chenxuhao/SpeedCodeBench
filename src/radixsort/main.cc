#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cassert>

void cpu_sort(unsigned int* h_out, unsigned int* h_in, size_t len) {
  for (size_t i = 0; i < len; ++i)
    h_out[i] = h_in[i];
  std::sort(h_out, h_out + len);
}

void test_cpu_vs_gpu(unsigned int* h_in, unsigned int num_elems);
void radix_sort(unsigned int* const h_out, unsigned int* const h_in, unsigned int d_in_len);
 
int main(int argc, char* argv[]) {
  // Set up clock for timing comparisons
  srand(1);
  for (int i = 27; i < 28; ++i) {
    assert(i < 31);
    unsigned int num_elems = (1 << i);
    //unsigned int num_elems = 8;
    std::cout << "h_in size: " << num_elems << std::endl;
    unsigned int* h_in = new unsigned int[num_elems];
    unsigned int* h_in_rand = new unsigned int[num_elems];
    for (unsigned int j = 0; j < num_elems; j++) {
      h_in[j] = (num_elems - 1) - j;
      h_in_rand[j] = rand() % num_elems;
    }
    std::cout << "*** i: " << i << " ***" << std::endl;
    for (int j = 0; j < 5; ++j) {
      std::cout << "*****Descending order*****" << std::endl;
      test_cpu_vs_gpu(h_in, num_elems);
      std::cout << "*****Random order*****" << std::endl;
      test_cpu_vs_gpu(h_in_rand, num_elems);
      std::cout << std::endl;
    }
    delete[] h_in;
    delete[] h_in_rand;
    std::cout << std::endl;
  }
}

void test_cpu_vs_gpu(unsigned int* h_in, unsigned int num_elems) {
  std::clock_t start;
  unsigned int* h_out_cpu = new unsigned int[num_elems];
  unsigned int* h_out_gpu = new unsigned int[num_elems];
  start = std::clock();
  cpu_sort(h_out_cpu, h_in, num_elems);
  double cpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  std::cout << "CPU time: " << cpu_duration << " s" << std::endl;
  start = std::clock();
  radix_sort(h_out_gpu, h_in, num_elems);
  double gpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  std::cout << "GPU time: " << gpu_duration << " s" << std::endl;
  // Calculate GPU / CPU speedup
  std::cout << "Speedup: " << cpu_duration / gpu_duration << "x" << std::endl;
  // Check for any mismatches between outputs of CPU and GPU
  bool match = true;
  int index_diff = 0;
  for (unsigned int i = 0; i < num_elems; ++i) {
    if (h_out_cpu[i] != h_out_gpu[i]) {
      match = false;
      index_diff = i;
      break;
    }
  }
  std::cout << "Match: " << match << std::endl;

  // Detail the mismatch if any
  if (!match) {
    std::cout << "Difference in index: " << index_diff << std::endl;
    std::cout << "CPU: " << h_out_cpu[index_diff] << std::endl;
    std::cout << "GPU Radix Sort: " << h_out_gpu[index_diff] << std::endl;
    int window_sz = 10;
    std::cout << "Contents: " << std::endl;
    std::cout << "CPU: ";
    for (int i = -(window_sz / 2); i < (window_sz / 2); ++i) {
      std::cout << h_out_cpu[index_diff + i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "GPU Radix Sort: ";
    for (int i = -(window_sz / 2); i < (window_sz / 2); ++i) {
      std::cout << h_out_gpu[index_diff + i] << ", ";
    }
    std::cout << std::endl;
  }
  delete[] h_out_gpu;
  delete[] h_out_cpu;
}

