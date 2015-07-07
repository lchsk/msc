#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// #include <mkl.h>
#include <time.h>

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

float**
new_matrix(int size)
{
    float** tmp = malloc (sizeof (float*) * size);

    for (int i = 0; i < size; i++)
        tmp[i] = malloc (sizeof (float) * size);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            tmp[i][j] = 0.0;

    return tmp;
}

float*
new_1d_matrix(int size)
{
    float* tmp = malloc (sizeof (float*) * size * size);

    for (int i = 0; i < size * size; i++)
        tmp[i] = 0.0;

    return tmp;
}

void
free_matrices(int size, float** A, float** B, float** C)
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
free_1d_matrices (float* A, float* B, float* C)
{
    if (A != NULL)
        free (A);

    if (B != NULL)
        free (B);

    if (C != NULL)
        free (C);
}

void
init_matrix(int size, float** m)
{
    #pragma omp parallel for default(none) shared(m, size)
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            m[i][j] = (float) (rand() % 10);
}

void
init_1d_matrix(int size, float* m)
{
    #pragma omp parallel for default(none) shared(m, size)
    for (int i = 0; i < size * size; ++i)
        m[i] = (float) (rand() % 10);
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
print_matrix (int size, float** m)
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
print_1d_matrix (int size, float* m)
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
is_correct_2d (int size, float** A, float** B, float**C)
{
    float** R = new_matrix (size);

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

// IKJ version
void
m_ikj(int size, float** A, float** B, float** C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
            for(int j = 0; j < size; j++)
                C[i][j] += A[i][k] * B[k][j];
}

// IKJ + restrict version
void
m_ikj_restrict(int size, float** restrict A, float** restrict B, float** restrict C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
            for(int j = 0; j < size; j++)
                C[i][j] += A[i][k] * B[k][j];
}

// IKJ + restrict + tmp version
void
m_ikj_restrict_tmp(int size, float** restrict A, float** restrict B, float** restrict C)
{
    float* restrict tmp;

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
m_ijk(int size, float** A, float** B, float** C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// IJK + restrict version
void
m_ijk_restrict(int size, float** restrict A, float** restrict B, float** restrict C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                C[i][j] +=  A[i][k] * B[k][j];
}

// m_vect_2d
void
m_vect_2d(int size, float** A, float** B, float** C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        float* restrict r = C[i];
        float* restrict u = A[i];

        for (int k = 0; k < size; ++k)
        {
            float* restrict v = B[k];

            for (int j = 0; j < size; ++j)
                r[j] += u[k] * v[j];
        }
    }
}

// ikj 1d notation
void
m_ikj_1d(int size, float* A, float* B, float* C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        float*  r = C + i * size;
        float*  u = A + i * size;

        for (int k = 0; k < size; ++k)
        {
            float*  v = B + k * size;

            for (int j = 0; j < size; ++j)
                r[j] += u[k] * v[j];
        }
    }
}

// array notation
void
m_array_not(int size, float* A, float* B, float* C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        for (int k = 0; k < size; ++k)
            C[i * size:size] += A[i * size + k] * B[k * size:size];
    }
}

__attribute__((vector))void mul_vect(float* a, float* b, float* c)
{
    // c[0] = a[0] * b[0];
    *c += *a * *b;
    return;
}

// elemental function
void
m_elem_fun(int size, float** A, float** B, float** C)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        float* r = C[i];
        float* u = A[i];

        for (int k = 0; k < size; ++k)
        {
            float* v = B[k];

            mul_vect(&u[0:size], &v[0:size], &r[0:size]);
            // for (int j = 0; j < size; ++j)
                // r[j] += u[k] * v[j];

        }
    }
}

void
m_ikj_unroll(int size, float** A, float** B, float** C)
{
	int uf = 4;
    float* restrict tmp;

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
m_tiling(int size, float* A, float* B, float* C)
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

    float** A = NULL;
    float** B = NULL;
    float** C = NULL;

    float* A1 = NULL;
    float* B1 = NULL;
    float* C1 = NULL;

    // ----------------------------------------------

    // IJK

    A = new_matrix(size);
    B = new_matrix(size);
    C = new_matrix(size);
    init_matrix(size, A);
    init_matrix(size, B);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_ijk(size, A, B, C);
    }

    print_results ("IJK", t_start, size, iter);

    free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IJK restrict

    A = new_matrix(size);
    B = new_matrix(size);
    C = new_matrix(size);
    init_matrix(size, A);
    init_matrix(size, B);

    t_start = omp_get_wtime();

    for (int idx = 0; idx < iter; idx++)
    {
        m_ijk_restrict(size, A, B, C);
    }

    print_results ("IJK restrict", t_start, size, iter);
    free_matrices (size, A, B, C);

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

    // print_matrix (size, A);
    // print_matrix (size, B);
    // print_matrix (size, C);

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

    // Array notation 2d

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

    free_matrices (size, A, B, C);

    return 0;


    // #pragma omp parallel for default(none) shared(A, B, C, size) private(r)
    // for (int i = 0; i < size; i++)
    //     for(int j = 0; j < size; j++)
    //         for (int k = 0; k < size; k++)
    //             C[i][j] += A[i][k] * B[k][j];
    // for (int i = 0; i < size; i++)
    //     for (int k = 0; k < size; k++)
    //         for(int j = 0; j < size; j++)
    //             C[i][j] += A[i][k] * B[k][j];



//
        // for (int i = 0; i < size; ++i)
        //     for (int k = 0; k < size; ++k)
        //         for (int j = 0; j < size; ++j)
        //
}
