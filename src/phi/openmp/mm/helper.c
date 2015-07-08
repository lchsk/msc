#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "helper.h"

DTYPE**
new_matrix(int size)
{
    DTYPE** tmp = malloc (sizeof (DTYPE*) * size);

    for (int i = 0; i < size; i++)
        tmp[i] = malloc (sizeof (DTYPE) * size);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            tmp[i][j] = 0.0;

    return tmp;
}

DTYPE*
new_1d_matrix(int size)
{
    DTYPE* tmp = malloc (sizeof (DTYPE*) * size * size);

    for (int i = 0; i < size * size; i++)
        tmp[i] = 0.0;

    return tmp;
}

void
free_matrices(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
    if (A != NULL)
    {
        for (int i = 0; i < size; i++)
            free (A[i]);
        free (A);
    }

    if (B != NULL)
    {
        for (int i = 0; i < size; i++)
            free (B[i]);
        free (B);
    }

    if (C != NULL)
    {
        for (int i = 0; i < size; i++)
            free (C[i]);
        free (C);
    }
}

void
free_1d_matrices (DTYPE* A, DTYPE* B, DTYPE* C)
{
    if (A != NULL)
        free (A);

    if (B != NULL)
        free (B);

    if (C != NULL)
        free (C);
}

void
init_matrix(int size, DTYPE** m)
{
    #pragma omp parallel for default(none) shared(m, size)
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            m[i][j] = (DTYPE) (rand() % 10);
}

void
init_1d_matrix(int size, DTYPE* m)
{
    #pragma omp parallel for default(none) shared(m, size)
    for (int i = 0; i < size * size; ++i)
        m[i] = (DTYPE) (rand() % 10);
}

void
print_results (char* str, double t_start, int size, int iter)
{
    // t_end = omp_get_wtime();
    double t_avg = (omp_get_wtime() - t_start) / iter;

    printf ("%s\n", str);
    printf ("\tTime: %f s\n", t_avg);
    printf ("\tGflops: %g\n", 2e-9 * size * size * size / t_avg);
}

void
print_matrix (int size, DTYPE** m)
{
    printf ("\n");

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            printf ("%.2f ", m[i][j]);

        printf ("\n");
    }

    printf ("\n");
}

void
print_1d_matrix (int size, DTYPE* m)
{
    printf ("\n");

    for (int i = 0; i < size * size; i++)
    {
        printf ("%.2f ", m[i]);

        if (i % size == size - 1)
            printf ("\n");
    }

    printf ("\n");
}

int
is_correct_2d (int size, DTYPE** A, DTYPE** B, DTYPE**C)
{
    DTYPE** R = new_matrix (size);

    #pragma omp parallel for default(none) shared(A, B, R, size)
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
            for(int j = 0; j < size; j++)
                R[i][j] += A[i][k] * B[k][j];

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            if (abs (C[i][j] - R[i][j]) > 0.01)
                return 0;

    return 1;
}
