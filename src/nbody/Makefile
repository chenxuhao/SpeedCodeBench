include ../common.mk

OBJS = main.o

all: bh_omp_base

bh_omp_base: $(OBJS) omp_base.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp -lm
	mv $@ $(BIN)

bh_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ -lgomp $(NVLIBS) -lm
	mv $@ $(BIN)

clean:
	rm *.o
