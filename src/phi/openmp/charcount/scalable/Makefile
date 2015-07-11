CC = icc
DEBUG =
CFLAGS = -O3 -Ofast -opt-assume-safe-padding -fno-alias

LIBS = -fopenmp
LIBS_MIC = -fopenmp

STD = -std=c99

FILES = cc.c

OUT = cc
OUT_MIC = miccc

all: $(OUT)

everything: $(OUT) $(OUT_MIC)

$(OUT):
	$(CC) $(FILES) -o $(OUT) $(CFLAGS) $(LIBS) $(STD)

$(OUT_MIC):
		$(CC) $(FILES) -o $(OUT_MIC) $(CFLAGS) $(LIBS_MIC) -mmic $(STD)

report:
	$(CC) $(FILES) -o $(OUT) $(CFLAGS) $(LIBS) $(STD) -vec-report6

report-mic:
	$(CC) $(FILES) -o $(OUT_MIC) $(CFLAGS) $(LIBS_MIC) -mmic $(STD) -vec-report6

clean:
	rm -rf *o *optrpt $(OUT) $(OUT_MIC)
