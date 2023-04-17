KERNELS = bc bfs blackscholes cc cf fft histogram kmeans lbm nbody pagerank \
		  reduction saxpy scan sgemm sssp spmv stencil streamcluster tc vc

.PHONY: all
all: $(KERNELS)

% : src/%/Makefile
	cd src/$@; make; cd ../..

.PHONY: clean
clean:
	rm src/*/*.o
