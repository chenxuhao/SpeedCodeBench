#include <fstream>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include "const.h"

int main(int argc, char* argv[]) {
  if(argc != 3) {
    std::cerr << "Usage: ./invkin <input file coefficients> <iterations>" << std::endl;
    exit(EXIT_FAILURE);
  }

  float* xTarget_in_h;
  float* yTarget_in_h;
  float* angle_out_h;
  float* angle_out_cpu;
  int data_size = 0;

  // process the files
  std::ifstream coordinate_in_file (argv[1]);
  const int iteration = atoi(argv[2]);

  if(coordinate_in_file.is_open()) {
    coordinate_in_file >> data_size;
    std::cout << "# Data Size = " << data_size << std::endl;
  }

  // allocate the memory
  xTarget_in_h = new (std::nothrow) float[data_size];
  if(xTarget_in_h == NULL) {
    std::cerr << "Memory allocation fails!!!" << std::endl;
    exit(EXIT_FAILURE);  
  }
  yTarget_in_h = new (std::nothrow) float[data_size];
  if(yTarget_in_h == NULL) {
    std::cerr << "Memory allocation fails!!!" << std::endl;
    exit(EXIT_FAILURE);  
  }
  angle_out_h = new (std::nothrow) float[data_size*NUM_JOINTS];
  if(angle_out_h == NULL) {
    std::cerr << "Memory allocation fails!!!" << std::endl;
    exit(EXIT_FAILURE);  
  }
  angle_out_cpu = new (std::nothrow) float[data_size*NUM_JOINTS];
  if(angle_out_cpu == NULL) {
    std::cerr << "Memory allocation fails!!!" << std::endl;
    exit(EXIT_FAILURE);  
  }

  // add data to the arrays
  float xTarget_tmp, yTarget_tmp;
  int coeff_index = 0;
  while(coeff_index < data_size) {  
    coordinate_in_file >> xTarget_tmp >> yTarget_tmp;
    for(int i = 0; i < NUM_JOINTS ; i++) {
      angle_out_h[coeff_index * NUM_JOINTS + i] = 0.0;
    }
    xTarget_in_h[coeff_index] = xTarget_tmp;
    yTarget_in_h[coeff_index++] = yTarget_tmp;
  }

  std::cout << "# Coordinates are read from file..." << std::endl;

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < iteration; n++) {
    invkin_omp(xTarget_in_h, yTarget_in_h, angle_out_cpu, data_size);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-3f) / iteration << " (us)\n";

  // CPU
  invkin_cpu(xTarget_in_h, yTarget_in_h, angle_out_cpu, data_size);

  // Check
  int error = 0;
  for (int i = 0; i < data_size; i++) {
    for (int j = 0 ; j < NUM_JOINTS; j++) {
      if ( fabsf(angle_out_h[i * NUM_JOINTS + j] - angle_out_cpu[i * NUM_JOINTS + j]) > 1e-3 ) {
        error++;
        break;
      }
    } 
  }
  // close files
  coordinate_in_file.close();
  // de-allocate the memory
  delete[] xTarget_in_h;
  delete[] yTarget_in_h;
  delete[] angle_out_h;
  delete[] angle_out_cpu;
  if (error) 
    std::cout << "FAIL\n";
  else 
    std::cout << "PASS\n";
  return 0;
}
