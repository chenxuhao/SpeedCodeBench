include ../common.mk
OBJS = main.o 
all: reduce_omp_base

reduce_omp_base: $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
