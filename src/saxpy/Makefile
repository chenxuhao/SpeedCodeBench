include ../common.mk

C_SRCS = main.c
C_OBJS = $(C_SRCS:.c=.o)

CILK_SRCS = cilk_base.c
CILK_OBJS = $(CILK_SRCS:.c=.o)

all: saxpy_cpu_base saxpy_omp_base saxpy_cilk_base

$(CILK_OBJS): %.o: %.c
	$(CILKCC) $(CILKCFLAGS) -c $< -o $@

saxpy_cpu_base: $(C_OBJS) cpu_base.o 
	$(CC) $(CFLAGS) $(INCLUDES) $(C_OBJS) cpu_base.o -o $@
	mv $@ $(BIN)

saxpy_omp_base: $(C_OBJS) omp_base.o 
	$(CC) $(CFLAGS) $(INCLUDES) $(C_OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

saxpy_gpu_base: $(C_OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(C_OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

saxpy_cilk_base: $(C_OBJS) cilk_base.o 
	$(CILKCC) $(CILKCFLAGS) $(INCLUDES) $(CILK_INC) $(C_OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

saxpy_zera_base: $(C_OBJS) zera_base.o 
	$(ZERACXX) $(ZERACXXFLAGS) $(INCLUDES) $(CILK_INC) $(C_OBJS) zera_base.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
