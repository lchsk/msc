//
// IKJ version
// 

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// 700 500 400

#define A_ROWS 700
#define A_COLS 500
#define B_COLS 400

int main (int argc, char *argv[])
{
    srand(time(NULL));
    int th_id, nthreads, i, j, k, chunk, tmp;

    double a[A_ROWS][A_COLS], b[A_COLS][B_COLS], c[A_ROWS][B_COLS];

    double t;

    chunk = 5;

    #pragma omp parallel shared(a, b, c, nthreads, chunk) private(th_id, i, j, k)
    {
        th_id = omp_get_thread_num();

        if (th_id == 0)
        {
            if (argv[1] == NULL)
            {
                nthreads = omp_get_num_threads();
                printf("No argument... Setting number of threads to %d\n", nthreads);
                
            }
            else
                nthreads = atoi(argv[1]);

            printf("Matrix multiplication, %d threads\n", nthreads);
        }

        // Initialisation

        #pragma omp for schedule (static, chunk)
        for (i = 0; i < A_ROWS; i++)
            for (j = 0; j < A_COLS; j++)
              a[i][j] = rand() % 1000;

        #pragma omp for schedule (static, chunk)
        for (i = 0; i < A_COLS; i++)
            for (j = 0; j < B_COLS; j++)
                b[i][j] = rand() % 1000;
        
        #pragma omp for schedule (static, chunk)
        for (i = 0; i < A_ROWS; i++)
            for (j = 0; j < B_COLS; j++)
                c[i][j] = rand() % 1000;


        t = omp_get_wtime();

        #pragma omp for schedule (static, chunk)
        for (i = 0; i < A_ROWS; i++)
        {
            for (k = 0; k < A_COLS; k++)
            {
                tmp = a[i][k];

                for(j = 0; j < B_COLS; j++)
                {
                    c[i][j] += tmp * b[k][j];
                }
            }
        }
    }

    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);
}