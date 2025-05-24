include ../common.mk
OBJS = main.o BaseGraph.o VertexSet.o
all: tc_omp_base tc_cilk_base

tc_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_omp_simd: $(OBJS) omp_simd.o intersect.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_simd.o intersect.o -o $@ -lgomp
	mv $@ $(BIN)

tc_cilk_base: $(OBJS) cilk_base.o 
	$(CILKCXX) $(CILKCXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

tc_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

tc_zera_base: main.o zera_base.o VertexSet.o
	$(ZERACXX) $(ZERACXXFLAGS) $(INCLUDES) $(CILK_INC) main.o zera_base.o VertexSet.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
