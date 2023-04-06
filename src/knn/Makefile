include ../common.mk
OBJS = main.o 
all: knn_omp_base

knn_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@
	mv $@ $(BIN)

knn_omp_base: $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp -lm
	mv $@ $(BIN)

clean:
	rm *.o
