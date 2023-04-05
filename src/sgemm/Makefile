include ../common.mk
OBJS = io.o main.o
all: sgemm_base sgemm_omp_base sgemm_omp

sgemm_base: $(OBJS) sgemm_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) sgemm_base.o -o $@ -lm
	mv $@ $(BIN)

sgemm_omp_base: $(OBJS) sgemm_omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) sgemm_omp_base.o -o $@ -lm -lgomp
	mv $@ $(BIN)

sgemm_omp: $(OBJS) sgemm_omp.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) sgemm_omp.o -o $@ -lm -lgomp
	mv $@ $(BIN)

