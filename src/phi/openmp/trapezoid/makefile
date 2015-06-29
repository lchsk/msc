OBJS = main.o
CC = icpc
DEBUG = 
CFLAGS = $(DEBUG)
LIBS = -fopenmp -vec-report3 -lm -O3 -Ofast
OUT = trap

all: $(OUT)

trap: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(OUT) $(LIBS)

main.o: main.c
	$(CC) -c $(CFLAGS) main.c $(LIBS)

clean:
	rm -rf *o $(OUT)
