include ../common.mk
OBJS += verifier.o main.o
all: cc_omp_base cc_omp_afforest 

cc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

cc_omp_afforest: omp_afforest.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_afforest.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
