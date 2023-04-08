include ../common.mk
OBJS = main.o points2image.o
all: daphne_omp_base 

daphne_omp_base: $(OBJS) omp_base.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

daphne_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

checkdata:
ifeq ($(wildcard ./p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
else
	$(warning p2i_input.dat found.)
endif
ifeq ($(wildcard ./p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
else
	$(warning p2i_output.dat found.)
endif

clean:
	rm *.o
