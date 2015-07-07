OBJS = mm.o
OBJS2 = mm-ijk.o
CC = icpc
DEBUG =
CFLAGS = $(DEBUG) -O3 -Ofast
LIBS = -fopenmp
OUT = mm
OUT2 = mm-ijk

all: $(OUT) farber-host farber-off farber-mic

mm: $(OBJS) $(OBJS2)
	$(CC) $(CFLAGS) $(OBJS) -o $(OUT) $(LIBS)
	$(CC) $(CFLAGS) $(OBJS2) -o $(OUT2) $(LIBS)

farber-host:
	icc -mkl -O3 -no-offload -openmp -Wno-unknown-pragmas -std=c99 -vec-report3 \
    mm-farber.c -o farber-host

farber-off:
	icc -mkl -O3 -offload-build -Wno-unknown-pragmas -std=c99 -vec-report3 \
mm-farber.c -o farber-off

farber-mic:
	icc -mkl -O3 -mmic -openmp -L  /opt/intel/lib/mic -Wno-unknown-pragmas \
-std=c99 -vec-report3 mm-farber.c -o farber-mic -liomp5

mm.o: mm.c
	$(CC) -c $(CFLAGS) mm.c $(LIBS)

mm-ijk.o: mm-ijk.c
	$(CC) -c $(CFLAGS) mm-ijk.c $(LIBS)

clean:
	rm -rf *o $(OUT) $(OUT2) farber-*
