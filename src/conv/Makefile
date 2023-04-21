include ../common.mk
OBJS = main.o verifier.o
all: conv_cpu_base conv_omp_base 

conv_cpu_base: $(OBJS) cpu_base.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lgomp
	mv $@ $(BIN)

conv_omp_base: $(OBJS) omp_base.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

conv_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

conv_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
