N=100000
B=1000
../../bin/fft_cpu_base $N $B
../../bin/fft_omp_base $N $B
../../bin/fft_cilk_base $N $B
../../bin/fft_gpu_base $N $B
