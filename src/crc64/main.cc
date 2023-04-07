#define _XOPEN_SOURCE 600
#include <ctime>
#include <vector>
#include <iostream>
#include "CRC64.h"

using namespace std;

int main(int argc, char *argv[]) {
  int ntests = 10;
  if (argc > 1) ntests = atoi(argv[1]);
  int seed = 5;
  if (argc > 2) seed = atoi(argv[2]);
  int max_test_length = 2097152;
  if (argc > 3) max_test_length = atoi(argv[3]);
  cout << "Running " << ntests << " tests with seed " << seed << endl;
  srand48(seed);

#ifdef __bgp__
#define THE_CLOCK CLOCK_REALTIME
#else
#define THE_CLOCK CLOCK_THREAD_CPUTIME_ID
#endif

  double tot_time = 0, tot_bytes = 0;
  int ntest = 0;
  while (++ntest <= ntests) {
    cout << ntest << " ";
    size_t test_length = (size_t) (max_test_length*(drand48()+1));
    cout << test_length << " ";
    vector<unsigned char> buffer(test_length);
    for (size_t i = 0; i < test_length; ++i) {
      buffer[i] = (unsigned char) (255*drand48());
    }
    timespec b_start, b_end;
    clock_gettime(THE_CLOCK, &b_start);
    uint64_t cs = crc64_omp(&buffer[0], test_length);
    clock_gettime(THE_CLOCK, &b_end);
    double b_time = (b_end.tv_sec - b_start.tv_sec);
    b_time += 1e-9*(b_end.tv_nsec - b_start.tv_nsec);
    tot_time += b_time;
    tot_bytes += test_length;
    // Copy the buffer and append the check bytes.
    size_t tlend = 8;
    buffer.resize(test_length + tlend, 0);
    crc64_invert(cs, &buffer[test_length]);
    string pass("pass"), fail("fail");
    uint64_t csc = crc64(&buffer[0], test_length+tlend);
    cout << ((csc == (uint64_t) -1) ? pass : fail) << " ";
    size_t div_pt = (size_t) (test_length*drand48());
    uint64_t cs1 = crc64(&buffer[0], div_pt);
    uint64_t cs2 = crc64(&buffer[div_pt], test_length - div_pt);
    csc = crc64_combine(cs1, cs2, test_length - div_pt);
    cout << ((csc == cs) ? pass : fail);
    cout << endl;
  }
  cout << (tot_bytes/(1024*1024))/tot_time << " MB/s" << endl;
  return 0;
}
