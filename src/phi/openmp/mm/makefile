CC = icc
DEBUG =
CFLAGS = -O3 -Ofast -opt-assume-safe-padding -fno-alias

LIBS = -fopenmp -mkl
LIBS_MIC = -fopenmp -mkl

STD = -std=c99

FILES = helper.c basic.c experimental.c elemental.c fast.c matrix.c

OUT = matrix
OUT_MIC = matrix-mic

all: $(OUT) $(OUT_MIC)

$(OUT):
	$(CC) $(FILES) -o $(OUT) $(CFLAGS) $(LIBS) $(STD)

$(OUT_MIC):
		$(CC) $(FILES) -o $(OUT_MIC) $(CFLAGS) $(LIBS_MIC) -mmic $(STD)

report:
	$(CC) $(FILES) -o $(OUT) $(CFLAGS) $(LIBS) $(STD) -vec-report6

report-mic:
	$(CC) $(FILES) -o $(OUT_MIC) $(CFLAGS) $(LIBS_MIC) -mmic $(STD) -vec-report6

clean:
	rm -rf *o $(OUT) $(OUT_MIC)
