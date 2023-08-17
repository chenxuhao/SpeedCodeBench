include ../common.mk
OBJS += main.o verifier.o
all: mis_omp_base mis_serial

mis_serial: $(OBJS) serial.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) serial.o -o $@ -lgomp
	mv $@ $(BIN)

mis_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

mis_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

mis_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
