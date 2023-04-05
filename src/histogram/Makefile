include ../common.mk
OBJS = main.o util.o
all: histo_cpu_base histo_omp_base

histo_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lm
	mv $@ $(BIN)

histo_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lm -lgomp
	mv $@ $(BIN)

histo_omp: $(OBJS) omp.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp.o -o $@ -lm -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
