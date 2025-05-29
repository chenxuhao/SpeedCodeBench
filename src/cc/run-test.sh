DATAPATH=../../inputs
#DATASET=citeseer
DATASET=livej

echo "../../bin/cc_omp_base $DATAPATH/$DATASET/graph"
../../bin/cc_omp_base $DATAPATH/$DATASET/graph
../../bin/cc_omp_afforest $DATAPATH/$DATASET/graph
../../bin/cc_cilk_base $DATAPATH/$DATASET/graph
../../bin/cc_cilk_afforest $DATAPATH/$DATASET/graph
