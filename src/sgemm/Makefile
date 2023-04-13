include ../common.mk
OBJS = main.o
all: sgemm_cpu_base sgemm_omp_base

sgemm_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lm
	mv $@ $(BIN)

sgemm_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ -lm $(NVLIBS)
	mv $@ $(BIN)

sgemm_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lm -lgomp
	mv $@ $(BIN)

sgemm_cpu_trans: $(OBJS) cpu_trans.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_trans.o -o $@ -lm -lgomp
	mv $@ $(BIN)

sgemm_omp_trans: $(OBJS) omp_trans.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_trans.o -o $@ -lm -lgomp
	mv $@ $(BIN)

sgemm_gpu_tiled: $(OBJS) gpu_tiled.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_tiled.o -o $@ -lm $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
