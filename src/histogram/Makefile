include ../common.mk
OBJS = main.o util.o
all: histo_cpu_base histo_omp_base

histo_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lm
	mv $@ $(BIN)

histo_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lm -lgomp
	mv $@ $(BIN)

histo_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@ -lm
	mv $@ $(BIN)

histo_gpu_base: $(OBJS) gpu_base.o gpu_kernels_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o gpu_kernels_base.o -o $@ -lm $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
