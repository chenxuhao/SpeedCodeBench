#DATAPATH=../../inputs
DATAPATH=~/OpenCilk/speedcode-problems/sssp/problem_data/sssp
#DATASET=citeseer
#DATASET=web-Google
DATASET=twitter40-adj-ordered
DATASET=soc-LiveJournal1

#../../bin/sssp_cpu_base $DATAPATH/$DATASET/graph
echo "../../bin/sssp_omp_base $DATAPATH/$DATASET/graph"
../../bin/sssp_omp_base $DATAPATH/$DATASET/graph
echo "../../bin/sssp_omp_dstep $DATAPATH/$DATASET/graph"
../../bin/sssp_omp_dstep $DATAPATH/$DATASET/graph
echo "../../bin/sssp_cilk_base $DATAPATH/$DATASET/graph"
../../bin/sssp_cilk_base $DATAPATH/$DATASET/graph
echo "../../bin/sssp_cilk_dstep $DATAPATH/$DATASET/graph"
../../bin/sssp_cilk_dstep $DATAPATH/$DATASET/graph

