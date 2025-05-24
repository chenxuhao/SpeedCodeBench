include ../common.mk
OBJS = main.o 
all: reduce_cpu_base reduce_omp_base reduce_cilk_base

reduce_cpu_base: $(OBJS) cpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lgomp
	mv $@ $(BIN)

reduce_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

reduce_gpu_base: $(OBJS) gpu_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

reduce_cilk_base: $(OBJS) cilk_base.o 
	$(CILKCXX) $(CILKCXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

reduce_zera_base: $(OBJS) zera_base.o 
	$(ZERACXX) $(ZERACXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) zera_base.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
