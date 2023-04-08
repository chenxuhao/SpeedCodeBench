include ../common.mk

fft_cpu_base: main.o cpu_base.o
	$(CXX) $(INCLUDES) main.o cpu_base.o -o $@ -lgomp -lm
	mv $@ $(BIN)

clean:
	rm *.o
