include ../common.mk
OBJS = main.o
all: fft_cpu_base fft_gpu_base fft_gpu_shmem

fft_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lgomp -lm
	mv $@ $(BIN)

fft_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

fft_gpu_shmem: $(OBJS) gpu_shmem.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_shmem.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
