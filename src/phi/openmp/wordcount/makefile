OBJS = main.o
CC = g++
DEBUG = 
CFLAGS = $(DEBUG) -std=c++0x
LIBS = -fopenmp -lm
OUT = wc

all: $(OUT)

wc: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(OUT) $(LIBS)

main.o: main.cpp
	$(CC) -c $(CFLAGS) main.cpp $(LIBS)


clean:
	rm -rf *o $(OUT)