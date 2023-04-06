#../../bin/stencil_cpu_base 128 128 32 100 small/input/128x128x32.bin out.bin
../../bin/stencil_omp_base 128 128 32 100 small/input/128x128x32.bin out.bin
python2 compare-output out.bin default/output/128x128x32.out
