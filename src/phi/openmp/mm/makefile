CC = icc
DEBUG =
CFLAGS = -O3 -Ofast

LIBS = -fopenmp
LIBS_MIC = -fopenmp

STD = -std=c99

FILES = helper.c matrix.c

OUT = matrix
OUT_MIC = matrix-mic

all: $(OUT) $(OUT_MIC)

$(OUT):
	$(CC) $(FILES) -o $(OUT) $(CFLAGS) $(LIBS) $(STD)

$(OUT_MIC):
		$(CC) $(FILES) -o $(OUT_MIC) $(CFLAGS) $(LIBS_MIC) -mmic $(STD)

clean:
	rm -rf *o $(OUT) $(OUT_MIC)
