#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

// Length of the alphabet
#define LEN 26
#define ALIGN 64
#define SMALL_A 97

#ifdef __MIC__
#define MIC 1
#else
#define MIC 0
#endif

#if MIC
    #include <xmmintrin.h>
#endif

// __attribute__((vector))
// void cmp(int* count, char c, int letter)
// {
//     *count += (c == letter) ? 1 : 0;
// }

int main(int argc, char* argv[])
{
    printf ("Character count (2)\n");

    if (argc != 4)
    {
        printf ("Usage: \n\t%s path threads tile_size\n", argv[0]);
        return 1;
    }

    double t, t2, t3;
    int th_id;
    int nthreads;
    int c, c1, c2, c3, c4;
    c = 0;
    int letter = 0;
    int start = 0;

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

    int regions = (int)floor(nthreads / LEN);
    int block_size = (int)floor(fsize / regions);

    int factor = (int)floor(nthreads / LEN);

    int tile = block_size;

    if (atoi (argv[3]) != 0)
        tile = atoi (argv[3]);

    printf ("Threads: %d/%d, Regions: %d, Block: %d, Tile: %d, Factor: %d\n", regions * LEN, nthreads, regions, block_size, tile, factor);

    #pragma omp barrier

    t = omp_get_wtime();

    #pragma omp parallel default(none) shared(str, block_size, final_count, factor, fsize, tile) private(start, th_id, c, letter) num_threads(nthreads)
    {
        th_id = omp_get_thread_num();
        letter = (int)floor(th_id / factor) + SMALL_A;
        start = (th_id % factor) * block_size;
        c = 0;

        // printf("letter : %c, id: %d, s: %d\n", letter, th_id, start);

        for (int i = start; i < min (start + block_size, fsize); i += tile)
        {
            // printf ("tile %d\n", letter);
            // char* restrict tmp = &str[i];

            // #pragma vector aligned
            // printf ("MAX: %d %d\n", i + tile, i + block_size);

            #if MIC
                #pragma unroll(4)
                // _MM_HINT_T0
                // #pragma prefetch str:0:16
            #endif
            __assume_aligned(str, ALIGN);
            for (int j = i; j < min (i + tile, start + block_size); j++)
            // for (int j = 0; j < tile; j++)
            {
                // if (j == i)
                // {
                //     printf ("loop len: %d %d\n", i, min (i + tile, start + block_size));
                // }

                // c += (str[j] == letter) ? 1 : 0;
                if (str[j] == letter)
                // if (tmp[i] == letter)
                    c++;
                // if ((int)letter == (int)str[j])
                    // c;



                // c++;
                // printf ("%c ", str[j]);
            }
            // break;

            // printf ("EXIT: %d\n", j);
        }

        // #pragma omp barrier
        // #pragma omp critical
        #pragma omp atomic
        final_count[letter - SMALL_A] += c;
        // #pragma omp flush(c)
        // printf ("C: %d\n", c);
        // printf ("SIEMA: %c = %d\n", letter, c);
    }
    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);

    #pragma omp parallel
    #pragma omp master

    for (int i = 0; i < LEN; i++)
        printf("%c = %d\n", i + SMALL_A, final_count[i]);

    _mm_free (str);

    return 0;
}
