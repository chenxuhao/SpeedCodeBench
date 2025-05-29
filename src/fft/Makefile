include ../common.mk
OBJS = main.o
all: fft_cpu_base fft_omp_base fft_gpu_base fft_cilk_base

fft_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lgomp -lm
	mv $@ $(BIN)

fft_omp_base: $(OBJS) omp_base.o
	$(CXX) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp -lm
	mv $@ $(BIN)

fft_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

fft_gpu_shmem: $(OBJS) gpu_shmem.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_shmem.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

fft_cilk_base: $(OBJS) cilk_base.o 
	$(CILKCXX) $(CILKCXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

fft_zera_base: $(OBJS) zera_base.o 
	$(ZERACXX) $(ZERACXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) zera_base.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
