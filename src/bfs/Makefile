include ../common.mk
OBJS += verifier.o main.o
all: bfs_cpu_base bfs_omp_base bfs_omp_direction bfs_cilk_base bfs_cilk_direction

bfs_cpu_base: cpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@
	mv $@ $(BIN)

bfs_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_topo: omp_topo.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_topo.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_direction: omp_direction.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_direction.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

bfs_cilk_direction: $(OBJS) cilk_direction.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_direction.o -o $@
	mv $@ $(BIN)

# data-driven BFS baseline
bfs_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
