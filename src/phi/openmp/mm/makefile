OBJS = mm.o
OBJS2 = mm-ijk.o
CC = icc
DEBUG = 
CFLAGS = $(DEBUG) -O3 -Ofast
LIBS = -fopenmp
OUT = mm
OUT2 = mm-ijk

all: $(OUT)

mm: $(OBJS) $(OBJS2)
	$(CC) $(CFLAGS) $(OBJS) -o $(OUT) $(LIBS)
	$(CC) $(CFLAGS) $(OBJS2) -o $(OUT2) $(LIBS)

mm.o: mm.c
	$(CC) -c $(CFLAGS) mm.c $(LIBS)

mm-ijk.o: mm-ijk.c
	$(CC) -c $(CFLAGS) mm-ijk.c $(LIBS)

clean:
	rm -rf *o $(OUT) $(OUT2)
