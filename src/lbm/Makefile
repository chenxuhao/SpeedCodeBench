include ../common.mk
OBJS = main.o
all: lbm_base lbm_omp

lbm_base: $(OBJS) lbm_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) lbm_base.o -o $@ -lm
	mv $@ $(BIN)

lbm_omp: $(OBJS) lbm_omp.o
	$(CC) $(CFLAGS) -D_OPENMP $(INCLUDES) $(OBJS) lbm_omp.o -o $@ -lm
	mv $@ $(BIN)

clean:
	rm *.o
