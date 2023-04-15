include ../common.mk
OBJS = main.o
all: lbm_cpu_base lbm_omp_base

lbm_cpu_base: $(OBJS) cpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lm
	mv $@ $(BIN)

lbm_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) -D_OPENMP $(INCLUDES) $(OBJS) omp_base.o -o $@ -lm
	mv $@ $(BIN)

lbm_gpu_base: $(OBJS) gpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS) -lm
	mv $@ $(BIN)

clean:
	rm *.o
