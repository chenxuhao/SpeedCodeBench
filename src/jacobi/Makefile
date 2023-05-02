include ../common.mk
NAME = jacobi
OBJS = main.o
all: $(NAME)_cpu_base $(NAME)_omp_base

$(NAME)_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lgomp -lm
	mv $@ $(BIN)

$(NAME)_omp_base: $(OBJS) omp_base.o
	$(CXX) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp -lm
	mv $@ $(BIN)

$(NAME)_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

$(NAME)_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
