include ../common.mk
OBJS += main.o
all: spmv_omp_base

spmv_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
