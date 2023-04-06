include ../common.mk
OBJS = main.o ops.o
all: bplustree_omp_base

bplustree_cpu_base: $(OBJS) cpu_base.o
	$(CC) $(CFLAGS) $(OBJS) cpu_base.o -o $@
	mv $@ $(BIN)

bplustree_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) $(OBJS) omp_base.o -o $@ -lm
	mv $@ $(BIN)

clean:
	rm -f *.o
