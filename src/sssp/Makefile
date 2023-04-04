include ../common.mk
OBJS += verifier.o main.o
all: sssp_omp_base 

sssp_omp_base: omp_dstep.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_dstep.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
