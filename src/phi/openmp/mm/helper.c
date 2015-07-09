#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "helper.h"

ALIGN_CODE DTYPE**
new_matrix(int size)
{
    #ifdef USE_ALIGNMENT
        ALIGN_CODE DTYPE** tmp = MALLOC (sizeof (DTYPE*) * size, ALIGN);
    #else
        ALIGN_CODE DTYPE** tmp = MALLOC (sizeof (DTYPE*) * size);
    #endif

    for (int i = 0; i < size; i++)
        #ifdef USE_ALIGNMENT
            tmp[i] = MALLOC (sizeof (DTYPE) * size, ALIGN);
        #else
            tmp[i] = MALLOC (sizeof (DTYPE) * size);
        #endif

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            tmp[i][j] = 0.0;

    return tmp;
    // #else
    //     tmp = MALLOC (sizeof (DTYPE*) * size);
    //
    //     for (int i = 0; i < size; i++)
    //     {
    //         DTYPE tmp[i] = MALLOC (sizeof (DTYPE*) * size);
    //     }
    //     for (int i = 0; i < size; i++)
    //         for (int j = 0; j < size; j++)
    //             tmp[i][j] = 0.0;
    // #endif

    // for (int i = 0; i < size; i++)
    // {
    //     #ifdef USE_ALIGNMENT
    //         ALIGN_CODE DTYPE tmp[i] = MALLOC (sizeof (DTYPE*) * size, ALIGN);
    //     #else
    //         DTYPE tmp[i] = MALLOC (sizeof (DTYPE*) * size);
    //     #endif
    // }
}

ALIGN_CODE DTYPE*
new_1d_matrix(int size)
{
    #ifdef USE_ALIGNMENT
        ALIGN_CODE DTYPE* tmp = MALLOC (sizeof (DTYPE*) * size * size, ALIGN);
    #else
        DTYPE* tmp = MALLOC (sizeof (DTYPE*) * size * size);
    #endif

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
            FREE (A[i]);
        FREE (A);
    }

    if (B != NULL)
    {
        for (int i = 0; i < size; i++)
            FREE (B[i]);
        FREE (B);
    }

    if (C != NULL)
    {
        for (int i = 0; i < size; i++)
            FREE (C[i]);
        FREE (C);
    }
}

void
free_1d_matrices (DTYPE* A, DTYPE* B, DTYPE* C)
{
    if (A != NULL)
        FREE (A);

    if (B != NULL)
        FREE (B);

    if (C != NULL)
        FREE (C);
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
print_results (
    char* str, // name of the implementation
    double t_start, // wallclock value
    int size,  // matrix size
    int iter, // number of iterations of calculations
    int mem_alloc, // 1 - matrix represented in 1D, 2 - in 2D
    int is_correct, // 1 - check if results are correct
    DTYPE** A, DTYPE** B, DTYPE** C, // 2D matrices
    DTYPE* A1, DTYPE* B1, DTYPE* C1 // 1D matrices
)
{
    double t_avg = (omp_get_wtime() - t_start) / iter;

    printf ("%s\n", str);
    printf ("\tTime: %f s\n", t_avg);
    printf ("\tGflops: %g\n", 2e-9 * size * size * size / t_avg);

    if (is_correct)
        if (mem_alloc == 1)
            printf ("\tCorrect: %d\n", is_correct_1d (size, A1, B1, C1));
        else if (mem_alloc == 2)
            printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
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
is_correct_2d (int size, DTYPE** A, DTYPE** B, DTYPE** C)
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

int
is_correct_1d (int size, DTYPE* A, DTYPE* B, DTYPE* C)
{
    DTYPE** R = new_matrix (size);

    #pragma omp parallel for default(none) shared(A, B, R, size)
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
            for(int j = 0; j < size; j++)
                R[i][j] += *(A + i * size + k) * *(B + k * size + j);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            if (abs (*(C + i * size + j) - R[i][j]) > 0.01)
                return 0;

    return 1;
}
