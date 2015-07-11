#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>

#define OP(a,b) (((a) ^ (b)) & 1 ^ 1)
#define F(a, b) (((a) ^ (b)))
#define AP(a) (( ~a & ( a + ~0 ) ) >> 31)

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

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

    double t, t2, t3;
    int th_id;
    int nthreads;

    int count[26];
    int final_count[26];

    for (int i = 0; i < 26; i++)
        count[i] = final_count[i] = 0;

    t = omp_get_wtime();

    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* str = (char*) _mm_malloc(fsize + 1, 64);
    fread(str, fsize, 1, f);
    fclose(f);

    printf("File size: %d\n", fsize);

    #pragma omp parallel
    #pragma omp master
    nthreads = omp_get_num_threads();
    nthreads = atoi (argv[2]);
    #pragma omp barrier



    int regions = nthreads / 26;
    int block_size = fsize / regions;
    int c = 0;

    int factor = nthreads / 26;

    int letter;

    printf ("Threads: %d/%d, Regions: %d, Block: %d, Factor: %d\n", regions * 26, nthreads, regions, block_size, factor);

    t = omp_get_wtime();
    #pragma omp parallel shared(str, block_size, final_count) private(count, th_id, c, letter) num_threads(nthreads)
    {
        th_id = omp_get_thread_num();
        letter = th_id / factor + 97;
        int start = (th_id % factor) * block_size;

        // printf ("id: %d, letter: %d (%c), start: %d\n", th_id, letter, letter, start);

        // #pragma vector aligned
        #pragma ivdep
        __assume_aligned(str, 64);
        for (int i = start; i < start + block_size; i++)
        {
            // c = (int)((int)str[i] & letter);

            // c += OP (str[i], letter);
            // c = c + (int)-AP(F((int)str[i], letter)) ;

            // printf ("%d  %d  %d\n", str[i], letter, -AP(F((int)str[i], letter)));

            // int b = 0;
            // b = b + (int)-AP(F((int)str[i], letter));

            // printf ("RES: %d\n", b);
            // break;
            #pragma vector aligned
            c = (str[i] == letter) ? c + 1 : c;
            // if ((int) str[i] == letter)
                // c++;
        }

        // printf ("letter: %c, c: %d\n", letter, c);
        #pragma omp critical
        final_count[letter - 97] += c;
    }
    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);

    printf ("\n%d\n", th_id);
    for (int i = 0; i < 26; i++)
    {
        printf("%c = %d\n", i+97, final_count[i]);
    }


    _mm_free (str);

    return 0;
}
