include ../common.mk

OBJS = main.o

CILK_SRCS = cilk_base.c
CILK_OBJS = $(CILK_SRCS:.c=.o)

all: bs_omp_base bs_omp_simd bs_cilk_base

$(CILK_OBJS): %.o: %.c
	$(CILKCC) $(CILKCFLAGS) -c $< -o $@

bs_omp_base: $(OBJS) blackscholes.o omp_base.o
	$(CC) $(CFLAGS) $(OBJS) blackscholes.o omp_base.o -o $@ -lm
	mv $@ $(BIN)

bs_omp_simd: omp_base.o blackscholes_simd.o
	$(CC) $(CFLAGS) $(OBJS) -DENABLE_SIMD -DNCO=4 blackscholes_simd.o omp_base.o -o $@ -lm
	mv $@ $(BIN)

bs_gpu_base: $(OBJS) gpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

bs_cilk_base: $(OBJS) cilk_base.o blackscholes.o
	$(CLANG) $(CILKCFLAGS) $(CILK_INC) $(OBJS) blackscholes.o cilk_base.o -o $@ -lm
	mv $@ $(BIN)

clean:
	rm *.o
