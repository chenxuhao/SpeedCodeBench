include ../common.mk
OBJS = main.o 
all: stencil_cpu_base stencil_omp_base

stencil_cpu_base: $(OBJS) cpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@
	mv $@ $(BIN)

stencil_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

stencil_gpu_base: $(OBJS) gpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
