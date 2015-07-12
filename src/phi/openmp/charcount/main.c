#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

int main(int argc, char* argv[])
{
    printf ("Character count (1)\n");

    if (argc != 3)
    {
        printf ("Usage: \n\t%s path threads\n", argv[0]);
        return 1;
    }

    double t, t2, t3;
    int th_id;
    int nthreads;

    int count[26];
    int final_count[26];

    t = omp_get_wtime();

    FILE *f = fopen(argv[1], "rb");

    if ( ! f)
    {
        printf ("File %s does not exist\n", argv[1]);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    printf("File size: %d\n", fsize);

    char *string = (char*) malloc(fsize + 1);
    fread(string, fsize, 1, f);
    fclose(f);

    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);

    for (int i = 0; i < 26; i++)
        final_count[i] = count[i] = 0;

    nthreads = atoi (argv[2]);

    printf("Using %d threads\n", nthreads);

    int part_size = ceil(fsize / (float) nthreads);

    t = omp_get_wtime();

    #pragma omp parallel private(count, th_id, t2, t3) shared(final_count, string) num_threads(nthreads)
    {
        th_id = omp_get_thread_num();

        int start = th_id * part_size;
        int end = start + part_size;
        int m = min(end, (int) fsize);

        for (int i = 0; i < 26; i++)
            count[i] = 0;

        // #pragma omp for
        // for (int i = 0; i < fsize; i++)
        // #pragma simd
        #pragma ivdep
        for (int i = start; i < m; i++)
        // for (int i = th_id; i < fsize; i += nthreads)
        {
            int index = (int) string[i] - 97;
            count[index]++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < 26; i++)
            {
                final_count[i] += count[i];
            }
        }
    }

    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);

    for (int i = 0; i < 26; i++)
        printf("%c = %d\n", i+97, final_count[i]);


    return 0;
}
