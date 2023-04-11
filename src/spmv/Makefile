include ../common.mk
OBJS += main.o
all: spmv_omp_base

spmv_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
