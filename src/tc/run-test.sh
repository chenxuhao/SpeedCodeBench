DATASET=citeseer
DATASET=livej
../../bin/tc_omp_base ../../inputs/$DATASET/graph
../../bin/tc_cilk_base ../../inputs/$DATASET/graph
../../bin/tc_gpu_base ../../inputs/$DATASET/graph
../../bin/tc_zera_base ../../inputs/$DATASET/graph
