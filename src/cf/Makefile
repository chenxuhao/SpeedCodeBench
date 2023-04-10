include ../common.mk
OBJS += verifier.o main.o

ifneq ($(SHOW_ERROR),0)
NVFLAGS += -DCOMPUTE_ERROR
CXXFLAGS += -DCOMPUTE_ERROR
endif

all: cf_omp_base 

cf_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) gpu_base.o $(OBJS) -o $@ $(NVLIBS) 
	mv $@ $(BIN)

cf_gpu_warp: gpu_warp.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) gpu_warp.o $(OBJS) -o $@ $(NVLIBS) -lgomp
	mv $@ $(BIN)

cf_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
