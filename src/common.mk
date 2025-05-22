DEBUG ?= 0
BIN := ../../bin

CC := gcc
CXX := g++
NVCC := nvcc
MPICC := mpicc
MPICXX := mpicxx
CLANG := $(CILK_HOME)/bin/clang
CLANGXX := $(CILK_HOME)/bin/clang++

CFLAGS    := -Wall -fopenmp
CXXFLAGS  := -Wall -fopenmp -std=c++17
ICPCFLAGS := -Wall -qopenmp

GENCODE_SM30 := -gencode arch=compute_30,code=sm_30
GENCODE_SM35 := -gencode arch=compute_35,code=sm_35
GENCODE_SM37 := -gencode arch=compute_37,code=sm_37
GENCODE_SM50 := -gencode arch=compute_50,code=sm_50
GENCODE_SM52 := -gencode arch=compute_52,code=sm_52
GENCODE_SM60 := -gencode arch=compute_60,code=sm_60
GENCODE_SM70 := -gencode arch=compute_70,code=sm_70
GENCODE_SM75 := -gencode arch=compute_75,code=sm_75
GENCODE_SM80 := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_SM86 := -gencode arch=compute_86,code=sm_86
CUDA_ARCH := $(GENCODE_SM70)
NVFLAGS := $(CUDA_ARCH)
NVFLAGS += -Xptxas -v
#NVFLAGS += -std=c++17
NVFLAGS += -DUSE_GPU

NVLIBS = -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcuda -lcudart
MPI_LIBS = -L$(MPI_HOME)/lib -lmpi
#CILKFLAGS = -O3 -fopenmp=libiomp5 -fopencilk
CILKFLAGS = -fopencilk -std=c++17
CILK_INC = -I$(CILK_HOME)/include
CUINC = -I$(CUDA_HOME)/include
INCLUDES = -I../../include
VPATH += ../common
OBJS = VertexSet.o graph.o

ifeq ($(DEBUG), 1)
	CFLAGS += -g -O0
	CXXFLAGS += -g -O0
	ICPCFLAGS += -g -O0
	CILKFLAGS += -g -O0
	NVFLAGS += -G
else
	CFLAGS += -O3
	CXXFLAGS += -O3
	ICPCFLAGS += -O3
	CILKFLAGS += -O3
	NVFLAGS += -O3 -w
endif

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(CUINC) -c $<

%.o: %.cxx
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) -c $<

