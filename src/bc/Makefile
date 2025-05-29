include ../common.mk
OBJS = BaseGraph.o VertexSet.o verifier.o main.o
all: bc_omp_base bc_gpu_base bc_cilk_base

bc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

# data driven baseline
bc_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@  $(BIN)

bc_cilk_base: $(OBJS) cilk_base.o 
	$(CILKCXX) $(CILKCXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
