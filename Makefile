KERNELS = bc bfs blackscholes cc cf fft kmeans nbody pagerank sgemm sssp spmv stencil streamcluster symgs tc vc

.PHONY: all
all: $(KERNELS)

% : src/%/Makefile
	cd src/$@; make; cd ../..

.PHONY: clean
clean:
	rm src/*/*.o
