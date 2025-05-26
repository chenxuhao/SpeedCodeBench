include ../common.mk
OBJS = main.o util.o
all: histo_cpu_base histo_omp_base histo_omp_atomfree histo_cilk_base

histo_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lm
	mv $@ $(BIN)

histo_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lm -lgomp
	mv $@ $(BIN)

histo_omp_atomfree: $(OBJS) omp_atomfree.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_atomfree.o -o $@ -lm -lgomp
	mv $@ $(BIN)

histo_cilk_base: $(OBJS) cilk_base.o 
	$(CILKCXX) $(CILKCXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@ -lm
	mv $@ $(BIN)

histo_gpu_base: $(OBJS) gpu_base.o gpu_kernels_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o gpu_kernels_base.o -o $@ -lm $(NVLIBS)
	mv $@ $(BIN)

histo_zera_base: $(OBJS) zera_base.o 
	$(ZERACXX) $(ZERACXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) zera_base.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
