#DATAPATH=../../inputs
DATAPATH=~/OpenCilk/speedcode-problems/sssp/problem_data/sssp
#DATASET=citeseer
DATASET=web-Google

#../../bin/sssp_cpu_base $DATAPATH/$DATASET/graph
../../bin/sssp_omp_base $DATAPATH/$DATASET/graph
../../bin/sssp_omp_dstep $DATAPATH/$DATASET/graph
../../bin/sssp_cilk_base $DATAPATH/$DATASET/graph

