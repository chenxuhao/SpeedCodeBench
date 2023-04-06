include ../common.mk
OBJS += main.o
all: mst_gpu_base

mst_cpu_base: $(OBJS) cpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) cpu_base.o -o $@ -lgomp

mst_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS) -lgomp
	mv $@ $(BIN)

clean:
	rm *.o

