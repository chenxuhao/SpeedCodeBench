#include <chrono>
#include <iostream>
#include <omp.h>
#include "points2image.h"

void usage(char *exec) {
  std::cout << "Usage: \n" << exec << " [-p N]\nOptions:\n  -p N   executes N invocations in sequence,";
  std::cout << "before taking time and check the result.\n";
  std::cout << "         Default: N=1\n";
}

int main(int argc, char **argv) {
  // how many testcases should be executed in sequence (before checking for correctness)
  int pipelined = 1;
  if ((argc != 1) && (argc !=  3)) {
    usage(argv[0]);
    exit(2);
  }
  if (argc == 3) {
    if (strcmp(argv[1], "-p") != 0) {
      usage(argv[0]);
      exit(3);
    }
    errno = 0;
    pipelined = strtol(argv[2], NULL, 10);
    if (errno || (pipelined < 1) ) {
      usage(argv[0]);
      exit(4);
    }
    std::cout << "Invoking kernel " << pipelined << " time(s) per measure/checking step\n";
  }
  // read input data
  points2image driver;
  driver.init();
  double start_time = omp_get_wtime();
  // execute the kernel
  driver.run(pipelined);
  double end_time = omp_get_wtime();
  double elapsed_time = end_time - start_time;
  std::cout << "Elapsed time: "<< elapsed_time << " seconds, average time per testcase (#"
            << driver.testcases << "): " << elapsed_time / (double) driver.testcases
            << " seconds" << std::endl;
  // read the desired output and compare
  if (driver.check_output())
    std::cout << "PASS\n";
  else 
    std::cout << "FAIL\n";
  return 0;
}
