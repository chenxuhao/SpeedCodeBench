include ../common.mk
OBJS = main.o
all: kmeans_cpu_base kmeans_omp_base kmeans_cilk_base kmeans_gpu_base

kmeans_cpu_base: $(OBJS) cpu_base.o
	$(CC) $(CFLAGS) $(OBJS) cpu_base.o -o $@
	mv $@ $(BIN)

kmeans_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) $(OBJS) omp_base.o -o $@
	mv $@ $(BIN)

kmeans_gpu_base: $(OBJS) gpu_base.o
	$(CC) $(CFLAGS) $(OBJS) gpu_base.o -o $@ $(NVLIBS) -lm
	mv $@ $(BIN)

kmeans_cilk_base: $(OBJS) cilk_base.o 
	$(CILKCXX) $(CILKCXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

kmeans_zera_base: $(OBJS) zera_base.o 
	$(ZERACXX) $(ZERACXXFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) zera_base.o -o $@
	mv $@ $(BIN)

clean:
	rm -f *.o
