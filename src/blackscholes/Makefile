include ../common.mk

all: blackscholes blackscholes_simd

blackscholes:
	$(CC) $(CFLAGS) -DENABLE_OPENMP $(INCLUDES) blackscholes.c -o $@ -lm
	mv $@ $(BIN)

blackscholes_simd:
	$(CC) $(CFLAGS) -DENABLE_OPENMP -DNCO=4 $(INCLUDES) blackscholes.simd.c -o $@ -lm
	mv $@ $(BIN)
