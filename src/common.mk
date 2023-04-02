DEBUG ?= 0
BIN := ../../bin

CC := gcc
CXX := g++

INCLUDES = -I../../include
VPATH += ../common
OBJS=VertexSet.o graph.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

