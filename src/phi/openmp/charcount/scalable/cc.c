#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>

// #define OP(a,b) (((a) ^ (b)) & 1 ^ 1)
// #define F(a, b) (((a) ^ (b)))
// #define AP(a) (( ~a & ( a + ~0 ) ) >> 31)

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

// Length of the alphabet
#define LEN 26
#define ALIGN 64
#define SMALL_A 97

__attribute__((vector))
void cmp(int* count, char c, int letter)
{
    *count += (c == letter) ? 1 : 0;
}

int main(int argc, char* argv[])
{
    // char a = 'a';
    //
    // printf ("RES: %d\n", -AP(F('a', 97)));
    // printf ("RES: %d\n", -AP(F('b', 97)));
    // printf ("RES: %d\n", -AP(F('c', 97)));
    // printf ("RES: %d\n", -AP(F('d', 100)));
    // printf ("RES: %d\n", -AP(F('e', 101)));
    //
    // return 0;

    printf ("Character count (2)\n");

    if (argc != 3)
    {
        printf ("Usage: \n\t%s path threads\n", argv[0]);
        return 1;
    }

    double t, t2, t3;
    int th_id;
    int nthreads;
    int c, c1, c2, c3, c4;
    c = c1 = c2 = c3 = c4 = 0;
    int letter = 0;

    int count[LEN];
    int final_count[LEN];

    for (int i = 0; i < LEN; i++)
        count[i] = final_count[i] = 0;

    t = omp_get_wtime();

    FILE* f = fopen(argv[1], "rb");

    if ( ! f)
    {
        printf ("File %s does not exist\n", argv[1]);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    __attribute__((aligned(ALIGN))) char* restrict str = (char*) _mm_malloc(fsize + 1, ALIGN);
    fread(str, fsize, 1, f);
    fclose(f);

    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);

    printf("File size: %d\n", fsize);

    #pragma omp parallel
    #pragma omp master

    nthreads = atoi (argv[2]);

    if (nthreads < LEN)
    {
        nthreads = LEN;
        printf ("Number of threads must be >= %d\n", LEN);
    }

    int regions = nthreads / LEN;
    int block_size = fsize / regions;

    // private count (for each letter)
    int factor = nthreads / LEN;

    printf ("Threads: %d/%d, Regions: %d, Block: %d, Factor: %d\n", regions * LEN, nthreads, regions, block_size, factor);

    #pragma omp barrier

    t = omp_get_wtime();
    #pragma omp parallel shared(str, block_size, final_count) private(count, th_id, c, c1, c2, c3, c4, letter) num_threads(nthreads)
    {
        th_id = omp_get_thread_num();
        letter = th_id / factor + SMALL_A;
        int start = (th_id % factor) * block_size;

        __assume_aligned(str, ALIGN);
        #pragma unroll(4)
        for (int i = start; i < start + block_size; i++)
        {
            #pragma vector aligned
            c += (str[i] == letter) ? 1 : 0;
        }

        #pragma omp critical
        final_count[letter - SMALL_A] += c;
    }
    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);

    #pragma omp parallel
    #pragma omp master

    for (int i = 0; i < LEN; i++)
        printf("%c = %d\n", i + SMALL_A, final_count[i]);

    _mm_free (str);

    return 0;
}
