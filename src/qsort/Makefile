include ../common.mk
OBJS = main.o verifier.o
all: qsort_cpu_base qsort_omp_base

qsort_cpu_base: $(OBJS) cpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lgomp
	mv $@ $(BIN)

qsort_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

qsort_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

qsort_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS) -lm
	mv $@ $(BIN)

clean:
	rm *.o
