include ../common.mk
OBJS = main.o
all: kmeans_omp_base

kmeans_cpu_base: $(OBJS) cpu_base.o
	$(CC) $(CFLAGS) $(OBJS) cpu_base.o -o $@
	mv $@ $(BIN)

kmeans_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) $(OBJS) omp_base.o -o $@
	mv $@ $(BIN)

clean:
	rm -f *.o
