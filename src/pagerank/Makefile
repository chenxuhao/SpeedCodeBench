include ../common.mk
OBJS += verifier.o main.o
all: pr_omp_base pr_omp_push 

pr_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_push: omp_push.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_push.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
