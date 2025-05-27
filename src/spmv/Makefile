include ../common.mk
OBJS = main.o BaseGraph.o VertexSet.o
all: spmv_cpu_base spmv_omp_base spmv_cilk_base spmv_gpu_base

spmv_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@
	mv $@ $(BIN)

spmv_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_omp_unroll: $(OBJS) omp_unroll.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_unroll.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

spmv_cilk_base: $(OBJS) cilk_base.o 
	$(CILKCXX) $(CILKCXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

spmv_zera_base: main.o zera_base.o VertexSet.o
	$(ZERACXX) $(ZERACXXFLAGS) $(INCLUDES) $(CILK_INC) main.o zera_base.o VertexSet.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
