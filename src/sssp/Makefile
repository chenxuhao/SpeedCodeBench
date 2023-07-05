include ../common.mk
OBJS += verifier.o main.o
all: sssp_omp_base sssp_omp_dstep sssp_cilk_base sssp_cilk_dstep

sssp_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

sssp_omp_dstep: omp_dstep.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_dstep.o -o $@ -lgomp
	mv $@ $(BIN)

sssp_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

sssp_cilk_dstep: $(OBJS) cilk_dstep.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_dstep.o -o $@
	mv $@ $(BIN)

sssp_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
