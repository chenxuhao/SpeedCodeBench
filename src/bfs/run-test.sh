#DATAPATH=../../inputs
DATAPATH=~/OpenCilk/speedcode-problems/sssp/problem_data/sssp
#DATASET=citeseer
DATASET=web-Google
DATASET=twitter40-adj-ordered
DATASET=soc-LiveJournal1

echo "../../bin/bfs_cpu_base $DATAPATH/$DATASET/graph"
../../bin/bfs_cpu_base $DATAPATH/$DATASET/graph
echo "../../bin/bfs_omp_base $DATAPATH/$DATASET/graph"
../../bin/bfs_omp_base $DATAPATH/$DATASET/graph
echo "../../bin/bfs_omp_direction $DATAPATH/$DATASET/graph 0 1"
../../bin/bfs_omp_direction $DATAPATH/$DATASET/graph 0 1
echo "../../bin/bfs_cilk_base $DATAPATH/$DATASET/graph"
../../bin/bfs_cilk_base $DATAPATH/$DATASET/graph
echo "../../bin/bfs_cilk_direction $DATAPATH/$DATASET/graph 0 1"
../../bin/bfs_cilk_direction $DATAPATH/$DATASET/graph 0 1
