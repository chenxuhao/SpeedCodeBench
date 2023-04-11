include ../common.mk
OBJS += main.o color.o
INCLUDES += -I$(CUDA_HOME)/include
all: $(OBJS) symgs_omp_base

symgs_omp_base: $(OBJS) omp_base.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

symgs_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS) -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
