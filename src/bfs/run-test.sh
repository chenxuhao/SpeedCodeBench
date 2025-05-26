DATAPATH=../../inputs
#DATASET=citeseer
#DATASET=web-Google
DATASET=livej

echo "../../bin/bfs_cpu_base $DATAPATH/$DATASET/graph 0"
../../bin/bfs_cpu_base $DATAPATH/$DATASET/graph 0
echo "../../bin/bfs_omp_base $DATAPATH/$DATASET/graph 0"
../../bin/bfs_omp_base $DATAPATH/$DATASET/graph 0
echo "../../bin/bfs_omp_sliding $DATAPATH/$DATASET/graph 0"
../../bin/bfs_omp_sliding $DATAPATH/$DATASET/graph 0
echo "../../bin/bfs_omp_direction $DATAPATH/$DATASET/graph 0"
../../bin/bfs_omp_direction $DATAPATH/$DATASET/graph 0
echo "../../bin/bfs_cilk_base $DATAPATH/$DATASET/graph 0"
../../bin/bfs_cilk_base $DATAPATH/$DATASET/graph 0
echo "../../bin/bfs_cilk_direction $DATAPATH/$DATASET/graph 0"
../../bin/bfs_cilk_direction $DATAPATH/$DATASET/graph 0
