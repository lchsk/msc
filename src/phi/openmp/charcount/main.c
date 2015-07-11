#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

int main(int argc, char* argv[])
{
    double t, t2, t3;
    int th_id;
    int nthreads;
    // __declspec(aligned(64)) char a[1000];
    // char* str = "abdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsadabdfkaofdafppeopwqeasdkcmxzcmasdsasadasdasdasdasdasdsad";

    // omp_set_num_threads(4);

    int count[26];
    int final_count[26];

    // int size = strlen(str);

    // printf("Size: %d\n", size);

    t = omp_get_wtime();

    FILE *f = fopen("default.txt", "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    printf("File size: %d\n", fsize);

    char *string = (char*) malloc(fsize + 1);
    char *string2 = (char*) malloc(fsize + 1);
    fread(string, fsize, 1, f);
    fclose(f);

    for (int i = 0; i < fsize; i++)
        string2[i] = string[i];

    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);
    printf("Current Time: %f ms\n", omp_get_wtime());

    // printf("Number of chars in a file: %d\n", strlen(string));





    for (int i = 0; i < 26; i++)
    {
        final_count[i] = count[i] = 0;
    }
    // for (int i = 0; i < size; i++)
    // {
    //     printf("%c ", str[i]);
    // }

    if (argv[1] == NULL)
        nthreads = omp_get_max_threads();
    else
        nthreads = atoi(argv[1]);

    printf("Using %d threads\n", nthreads);

    int part_size = ceil(fsize / (float) nthreads);

    t = omp_get_wtime();

    #pragma omp sections
    {
        #pragma omp section
        {
            omp_set_num_threads(nthreads);

            #pragma omp parallel private(count, th_id, t2, t3) shared(final_count, string)
            {
                th_id = omp_get_thread_num();

                int start = th_id * part_size;
                int end = start + part_size;
                int m = min(end, (int) fsize);

                for (int i = 0; i < 26; i++)
                    count[i] = 0;

                // #pragma omp for
                // for (int i = 0; i < fsize; i++)
                // t3 = omp_get_wtime();
                // #pragma simd
                #pragma ivdep
                for (int i = start; i < m; i++)
                // for (int i = th_id; i < fsize; i += nthreads)
                {

                    // printf("%d ", str[i]);
                    int index = (int) string[i] - 97;
                    count[index]++;
                }
                // printf("Time for loop in T %d: %f ms\n", th_id, (omp_get_wtime() - t3) * 1000);

                #pragma omp critical
                {
                    t2 = omp_get_wtime();
                    for (int i = 0; i < 26; i++)
                    {
                        // printf("%c = %d\n", i+97, count[i]);
                        final_count[i] += count[i];
                    }
                    // printf("Time for critical section in T %d: %f ms\n", th_id, (omp_get_wtime() - t2) * 1000);
                }
            }
        }
    }



    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);
    printf("Current Time: %f ms\n", omp_get_wtime());

    for (int i = 0; i < 26; i++)
    {
        printf("%c = %d\n", i+97, final_count[i]);
    }

    // for (int i = 0; i < 26; i++)
    // {
    //     final_count[i] = count[i] = 0;
    // }
    //
    // free(string);
    //
    // int c = 0;
    // int factor = 2;
    //
    // t = omp_get_wtime();
    // #pragma omp parallel shared(string2) private(count, c, th_id, factor) num_threads(26)
    // {
    //     // for (int i = 0; i < fsize; i++)
    //     //     int t = string2[i];
    //     //
    //     // for (int i = 0; i < 26; i++)
    //     //     count[i] = 0;
    //
    //     th_id = omp_get_thread_num();
    //     int letter = th_id + 97;
    //
    //     for (int i = 0; i < fsize; i++)
    //     {
    //         if ((int) string2[i] == letter)
    //             c++;
    //             // count[th_id]++;
    //     }
    //
    //     // printf ("C: %d\n", c);
    //     final_count[th_id] = c;
    //
    //     // #pragma omp critical
    //     // {
    //     //     // for (int i = 0; i < 26; i++)
    //     //         // final_count[i] = 0;
    //     //
    //     //     for (int i = 0; i < 26; i++)
    //     //     {
    //     //         // final_count[i] += c;
    //     //         // final_count[i] += count[th_id];
    //     //     }
    //     // }
    // }
    // printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);
    //
    // printf ("\n%d\n", th_id);
    // for (int i = 0; i < 26; i++)
    // {
    //     printf("%c = %d\n", i+97, final_count[i]);
    // }




    free(string2);

    return 0;
}
