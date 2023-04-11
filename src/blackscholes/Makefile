include ../common.mk

OBJS = main.o blackscholes.o

all: blackscholes blackscholes_simd

blackscholes: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) -DENABLE_OPENMP $(INCLUDES) $(OBJS) omp_base.o -o $@ -lm
	mv $@ $(BIN)

blackscholes_simd:
	$(CC) $(CFLAGS) -DENABLE_OPENMP -DNCO=4 $(INCLUDES) blackscholes.simd.c -o $@ -lm
	mv $@ $(BIN)
