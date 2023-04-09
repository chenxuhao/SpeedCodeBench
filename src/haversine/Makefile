include ../common.mk
OBJS = main.o
all: haversine_omp_base haversine_gpu_base

haversine_omp_base: $(OBJS) omp_base.o
	$(CXX) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp -lm
	mv $@ $(BIN)

haversine_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
