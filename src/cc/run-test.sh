#DATAPATH=../../inputs
DATAPATH=~/OpenCilk/speedcode-problems/triangle-counting/problem_data/triangle-counting
#DATASET=citeseer
DATASET=livej

../../bin/cc_omp_base $DATAPATH/$DATASET/graph
../../bin/cc_omp_afforest $DATAPATH/$DATASET/graph
../../bin/cc_cilk_base $DATAPATH/$DATASET/graph
../../bin/cc_cilk_afforest $DATAPATH/$DATASET/graph
