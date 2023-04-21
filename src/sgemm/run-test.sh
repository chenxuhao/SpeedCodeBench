../../bin/sgemm_cpu_base small/input/matrix1.txt small/input/matrix2.txt small/input/matrix2t.txt out-cpu.txt
python2 compare-output small/output/matrix3.txt out-cpu.txt 
../../bin/sgemm_omp_base small/input/matrix1.txt small/input/matrix2.txt small/input/matrix2t.txt out-omp.txt
python2 compare-output small/output/matrix3.txt out-omp.txt 
