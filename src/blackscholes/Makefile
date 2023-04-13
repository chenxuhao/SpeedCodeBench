include ../common.mk

OBJS = main.o

all: bs_omp_base bs_omp_simd

bs_omp_base: $(OBJS) blackscholes.o omp_base.o
	$(CC) $(CFLAGS) $(OBJS) blackscholes.o omp_base.o -o $@ -lm
	mv $@ $(BIN)

bs_omp_simd: omp_base.o blackscholes_simd.o
	$(CC) $(CFLAGS) $(OBJS) -DENABLE_SIMD -DNCO=4 blackscholes_simd.o omp_base.o -o $@ -lm
	mv $@ $(BIN)

clean:
	rm *.o
