include ../common.mk
OBJS = main.o 
all: reduce_omp_base

reduce_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

reduce_gpu_base: $(OBJS) gpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
