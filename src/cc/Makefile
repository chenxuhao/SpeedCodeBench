include ../common.mk
OBJS += verifier.o main.o
all: cc_omp_base cc_omp_afforest cc_cilk_base cc_cilk_afforest

cc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

cc_omp_afforest: omp_afforest.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_afforest.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

cc_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

cc_cilk_afforest: $(OBJS) cilk_afforest.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_afforest.o -o $@
	mv $@ $(BIN)

cc_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

cc_gpu_warp: gpu_warp.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_warp.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

cc_gpu_afforest: gpu_afforest.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_afforest.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
