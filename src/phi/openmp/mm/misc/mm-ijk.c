//
// IJK version
//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define A_ROWS 700
#define A_COLS 500
#define B_COLS 400

int main (int argc, char *argv[])
{
    srand(time(NULL));
    int th_id, nthreads, i, j, k, tmp;

    double a[A_ROWS][A_COLS], b[A_COLS][B_COLS], c[A_ROWS][B_COLS];

    // Initialisation

    for (i = 0; i < A_ROWS; i++)
        for (j = 0; j < A_COLS; j++)
          a[i][j] = rand() % 1000;

    for (i = 0; i < A_COLS; i++)
        for (j = 0; j < B_COLS; j++)
            b[i][j] = rand() % 1000;
    
    for (i = 0; i < A_ROWS; i++)
        for (j = 0; j < B_COLS; j++)
            c[i][j] = rand() % 1000;

    double t;
    t = omp_get_wtime();

    #pragma omp parallel shared(a, b, c, nthreads) private(th_id, i, j, k)
    {
        #pragma omp for schedule (dynamic)
        for (i = 0; i < A_ROWS; i++)
            for(j = 0; j < B_COLS; j++)
                for (k = 0; k < A_COLS; k++)
                    c[i][j] += a[i][k] * b[k][j];
    }

    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);
}