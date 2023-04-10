include ../common.mk
OBJS += verifier.o main.o
all: bc_omp_base

bc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

# data driven baseline
bc_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@  $(BIN)

clean:
	rm *.o
