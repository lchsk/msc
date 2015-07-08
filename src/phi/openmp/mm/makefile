OBJS = mm.o
OBJS2 = mm-ijk.o
CC = icc
DEBUG =
CFLAGS = $(DEBUG) -O3 -Ofast
LIBS = -fopenmp
OUT = mm
OUT2 = mm-ijk

all: mm

mm:
	$(CC) helper.c matrix.c -o matrix -O2 -fopenmp -std=c99

clean:
	rm -rf *o
