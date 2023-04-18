include ../common.mk
INCLUDES += -I./cpu_headers -I./gpu_headers

all: rt_cpu_base

rt_cpu_base: main.o cpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o cpu_base.o -o $@
	mv $@ $(BIN)

rt_gpu_base: main.o gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

out.ppm: rt
	rm -f out.ppm
	time ./rt_cpu_base > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

clean:
	rm -f *.o *.ppm *.jpg
