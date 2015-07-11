OBJS = main.o
CC = icpc
DEBUG = -opt-report-phase=hlo -vec-report3
CFLAGS = $(DEBUG)
LIBS = -fopenmp -lm -O3 -Ofast
OUT = cc

all: $(OUT)

cc: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(OUT) $(LIBS)

main.o: main.c
	$(CC) -c $(CFLAGS) main.c $(LIBS)

clean:
	rm -rf *o $(OUT)
