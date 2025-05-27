include ../common.mk
OBJS = BaseGraph.o VertexSet.o verifier.o main.o
all: pr_omp_base pr_omp_push pr_cilk_base

pr_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_push: omp_push.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_push.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

pr_cilk_base: $(OBJS) cilk_base.o 
	$(CILKCXX) $(CILKCXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

pr_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

pr_gpu_warp: gpu_warp.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_warp.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

pr_zera_base: main.o zera_base.o VertexSet.o verifier.o
	$(ZERACXX) $(ZERACXXFLAGS) $(INCLUDES) $(CILK_INC) main.o zera_base.o VertexSet.o verifier.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
