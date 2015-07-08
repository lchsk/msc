#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// #include <mkl.h>
#include <time.h>

#include "helper.h"

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))




// IKJ version
void
m_ikj(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
            for(int j = 0; j < size; j++)
                C[i][j] += A[i][k] * B[k][j];
}

// IKJ + restrict version
void
m_ikj_restrict(int size, DTYPE** restrict A, DTYPE** restrict B, DTYPE** restrict C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
            for(int j = 0; j < size; j++)
                C[i][j] += A[i][k] * B[k][j];
}

// IKJ + restrict + tmp version
void
m_ikj_restrict_tmp(int size, DTYPE** restrict A, DTYPE** restrict B, DTYPE** restrict C)
{
    DTYPE* restrict tmp;

    #pragma omp parallel for default(none) shared(A, B, C, size) private(tmp)
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
        {
            tmp = &A[i][k];

            for(int j = 0; j < size; j++)
                C[i][j] += *tmp * B[k][j];
        }
}

// IJK version
void
m_ijk(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// IJK + restrict version
void
m_ijk_restrict(int size, DTYPE** restrict A, DTYPE** restrict B, DTYPE** restrict C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                C[i][j] +=  A[i][k] * B[k][j];
}

// m_vect_2d
void
m_vect_2d(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        DTYPE* restrict r = C[i];
        DTYPE* restrict u = A[i];

        for (int k = 0; k < size; ++k)
        {
            DTYPE* restrict v = B[k];

            for (int j = 0; j < size; ++j)
                r[j] += u[k] * v[j];
        }
    }
}

// ikj 1d notation
void
m_ikj_1d(int size, DTYPE* A, DTYPE* B, DTYPE* C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        DTYPE*  r = C + i * size;
        DTYPE*  u = A + i * size;

        for (int k = 0; k < size; ++k)
        {
            DTYPE*  v = B + k * size;

            for (int j = 0; j < size; ++j)
                r[j] += u[k] * v[j];
        }
    }
}

// array notation
void
m_array_not(int size, DTYPE* A, DTYPE* B, DTYPE* C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        for (int k = 0; k < size; ++k)
            C[i * size:size] += A[i * size + k] * B[k * size:size];
    }
}

__attribute__((vector))void mul_vect(DTYPE* a, DTYPE* b, DTYPE* c)
{
    // c[0] = a[0] * b[0];

    *c += *a * *b;
    // printf ("%.2f x %.2f = %.2f\n", *a, *b, *c);
    // return;
}

// elemental function
void
m_elem_fun(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        DTYPE* r = C[i];
        DTYPE* u = A[i];

        for (int k = 0; k < size; ++k)
        {
            DTYPE* v = B[k];

            mul_vect(&u[k], &v[0:size], &r[0:size]);

            // mul_vect(&u[0:size], &v[0:size], &r[0:size]);
            // for (int j = 0; j < size; ++j)
                // r[j] += u[k] * v[j];

        }
    }
}

void
m_ikj_unroll(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
	int uf = 4;
    DTYPE* restrict tmp;

    #pragma omp parallel for default(none) shared(A, B, C, size, uf) private(tmp)
	for (int i = 0; i < size; i++)
    {
		for (int k = 0; k < size; k++)
        {
			tmp = &A[i][k];

			for (int j = 0; j < size / uf; j += uf)
            {
				C[i][j]     += *tmp * B[k][j];
				C[i][j + 1] += *tmp * B[k][j + 1];
				C[i][j + 2] += *tmp * B[k][j + 2];
				C[i][j + 3] += *tmp * B[k][j + 3];
			}

			for (int j = size - uf; j < size; j++)
				C[i][j] += *tmp * B[k][j];
		}
	}
}

// tiling
void
m_tiling(int size, DTYPE* A, DTYPE* B, DTYPE* C)
{
    int tile_size = 16;
    #pragma omp parallel for default(none) shared(A, B, C, size, tile_size)
    for(int i = 0; i < size; i += tile_size)
        for(int j = 0; j < size; j += tile_size)
            for(int k = 0; k < size; k += tile_size)
                for(int ii = i; ii < min(i + tile_size, size); ++ii)
                    for(int jj = j; jj < min(j + tile_size, size); ++jj)
                        for(int kk = k; kk < min(k + tile_size, size); ++kk)
                            C[ii * size + jj] += A[ii * size + kk] + B[kk * size + jj];
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    int size = 1000;

    if (argv[1] != NULL)
        size = atoi (argv[1]);

    int threads = 4;
    int iter = 1;
    double t_avg = 0.0;
    double t_start;

    DTYPE** A = NULL;
    DTYPE** B = NULL;
    DTYPE** C = NULL;

    DTYPE* A1 = NULL;
    DTYPE* B1 = NULL;
    DTYPE* C1 = NULL;

    // ----------------------------------------------

    // IJK

    // A = new_matrix(size);
    // B = new_matrix(size);
    // C = new_matrix(size);
    // init_matrix(size, A);
    // init_matrix(size, B);
    //
    // t_start = omp_get_wtime();
    //
    // for (int idx = 0; idx < iter; idx++)
    // {
    //     m_ijk(size, A, B, C);
    // }
    //
    // print_results ("IJK", t_start, size, iter);
    //
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IJK restrict

    // A = new_matrix(size);
    // B = new_matrix(size);
    // C = new_matrix(size);
    // init_matrix(size, A);
    // init_matrix(size, B);
    //
    // t_start = omp_get_wtime();
    //
    // for (int idx = 0; idx < iter; idx++)
    // {
    //     m_ijk_restrict(size, A, B, C);
    // }
    //
    // print_results ("IJK restrict", t_start, size, iter);
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ

    A = new_matrix(size);
    B = new_matrix(size);
    C = new_matrix(size);
    init_matrix(size, A);
    init_matrix(size, B);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_ikj(size, A, B, C);
    }

    print_results ("IKJ", t_start, size, iter);
    free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ restrict

    A = new_matrix(size);
    B = new_matrix(size);
    C = new_matrix(size);
    init_matrix(size, A);
    init_matrix(size, B);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_ikj_restrict(size, A, B, C);
    }

    print_results ("IKJ restrict", t_start, size, iter);
    free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ restrict tmp

    A = new_matrix(size);
    B = new_matrix(size);
    C = new_matrix(size);
    init_matrix(size, A);
    init_matrix(size, B);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_ikj_restrict_tmp(size, A, B, C);
    }

    print_results ("IKJ restrict tmp", t_start, size, iter);
    free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ vect 2d

    A = new_matrix(size);
    B = new_matrix(size);
    C = new_matrix(size);
    init_matrix(size, A);
    init_matrix(size, B);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_vect_2d(size, A, B, C);
    }


    print_results ("IKJ vect 2d", t_start, size, iter);

    // print_matrix (size, A);
    // print_matrix (size, B);
    // print_matrix (size, C);


    free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ 1d notation

    A1 = new_1d_matrix(size);
    B1 = new_1d_matrix(size);
    C1 = new_1d_matrix(size);
    init_1d_matrix(size, A1);
    init_1d_matrix(size, B1);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_ikj_1d(size, A1, B1, C1);
    }

    print_results ("IKJ 1d notation", t_start, size, iter);

    free_1d_matrices (A1, B1, C1);

    // ----------------------------------------------

    // array notation

    A1 = new_1d_matrix(size);
    B1 = new_1d_matrix(size);
    C1 = new_1d_matrix(size);
    init_1d_matrix(size, A1);
    init_1d_matrix(size, B1);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_array_not(size, A1, B1, C1);
    }

    print_results ("Array notation", t_start, size, iter);

    free_1d_matrices (A1, B1, C1);

    // ----------------------------------------------

    // IKJ unroll

    A = new_matrix(size);
    B = new_matrix(size);
    C = new_matrix(size);
    init_matrix(size, A);
    init_matrix(size, B);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_ikj_unroll (size, A, B, C);
    }


    print_results ("IKJ unroll", t_start, size, iter);

    free_matrices (size, A, B, C);

    // ----------------------------------------------

    // tiling

    A1 = new_1d_matrix(size);
    B1 = new_1d_matrix(size);
    C1 = new_1d_matrix(size);
    init_1d_matrix(size, A1);
    init_1d_matrix(size, B1);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_tiling(size, A1, B1, C1);
    }

    print_results ("Tiling", t_start, size, iter);

    free_1d_matrices (A1, B1, C1);

    // ----------------------------------------------

    // Elem function

    A = new_matrix(size);
    B = new_matrix(size);
    C = new_matrix(size);
    init_matrix(size, A);
    init_matrix(size, B);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_elem_fun(size, A, B, C);
    }

    print_results ("Elemental function", t_start, size, iter);
    printf ("Correct: %d\n", is_correct_2d (size, A, B, C));


    // print_matrix (size, A);
    // print_matrix (size, B);
    // print_matrix (size, C);
    free_matrices (size, A, B, C);

    // DTYPE a[] = {1,2,3};
    // DTYPE b[] = {1,2,3};
    // DTYPE c[] = {0,0,0};

    // mul_vect(&a[:], &b[:], &c[:]);
    //
    // for (int i = 0; i < 3; i++)
    //     printf ("%f ", c[i]);

    return 0;
}
