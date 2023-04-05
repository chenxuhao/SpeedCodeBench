DEBUG ?= 0
BIN := ../../bin

CC := gcc
CXX := g++

CFLAGS    := -O3 -Wall -fopenmp
CXXFLAGS  := -Wall -fopenmp -std=c++17
ICPCFLAGS := -O3 -Wall -qopenmp

INCLUDES = -I../../include
VPATH += ../common
OBJS=VertexSet.o graph.o

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G
else
	CXXFLAGS += -O3
	NVFLAGS += -O3 -w
endif

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

