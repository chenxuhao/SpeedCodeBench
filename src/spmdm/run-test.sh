#DATAPATH=../../inputs
DATAPATH=~/OpenCilk/speedcode-problems/sssp/problem_data/sssp
DATASET=web-Google
DATASET=twitter40-adj-ordered
DATASET=citeseer
DATASET=soc-LiveJournal1
BENCH=spmdm

echo "../../bin/$BENCH\_cpu_base $DATAPATH/$DATASET/graph"
../../bin/$BENCH\_cpu_base $DATAPATH/$DATASET/graph
echo "../../bin/$BENCH\_omp_base $DATAPATH/$DATASET/graph"
../../bin/$BENCH\_omp_base $DATAPATH/$DATASET/graph
echo "../../bin/$BENCH\_cilk_base $DATAPATH/$DATASET/graph"
../../bin/$BENCH\_cilk_base $DATAPATH/$DATASET/graph

