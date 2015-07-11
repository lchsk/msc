#include "helper.h"
#include "experimental.h"
#include "elemental.h"
#include "fast.h"

#include <string.h>

#if USE_MKL
    #include "mkl.h"
#endif

void
m_test(int size, DTYPE* A, DTYPE* B, DTYPE* C, int tile_size)
{
    int i, j, k;
    int ROWCHUNK = tile_size;
    int COLCHUNK = tile_size;

    #pragma omp parallel for collapse(2) private(i,j,k)
    for(i = 0; i < size; i += ROWCHUNK)
    {
        for(j = 0; j < size; j += ROWCHUNK)
        {
            for(k = 0; k < size; k += COLCHUNK)
            {
                for (int ii = i; ii < min(i + ROWCHUNK, size); ii += 8)
                {
                    for (int kk = k; kk < min(k + COLCHUNK, size); kk++)
                    {
                        #pragma ivdep
                        #pragma vector aligned
	                    for (int jj = j; jj < min(j + ROWCHUNK, size); jj ++)
                        {
                            // mul_vect(&C[ii], &A[ii:8], &B[kk:8]);
                    		C[(ii*size)+jj] += A[(ii*size)+kk]*B[kk*size+jj];
                    		C[((ii+1)*size)+jj] += A[((ii+1)*size)+kk]*B[kk*size+jj];
                    		C[((ii+2)*size)+jj] += A[((ii+2)*size)+kk]*B[kk*size+jj];
                    		C[((ii+3)*size)+jj] += A[((ii+3)*size)+kk]*B[kk*size+jj];
                    		C[((ii+4)*size)+jj] += A[((ii+4)*size)+kk]*B[kk*size+jj];
                    		C[((ii+5)*size)+jj] += A[((ii+5)*size)+kk]*B[kk*size+jj];
                            C[((ii+6)*size)+jj] += A[((ii+6)*size)+kk]*B[kk*size+jj];
                            C[((ii+7)*size)+jj] += A[((ii+7)*size)+kk]*B[kk*size+jj];
                        }
                    }
                }
            }
        }
    }
}

void
m_test_2d(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size)
{
    int i, j, k;
    int ROWCHUNK = tile_size;
    int COLCHUNK = tile_size;

    #pragma omp parallel for collapse(2) private(i,j,k)
    for(i = 0; i < size; i += ROWCHUNK)
    {
        for(j = 0; j < size; j += ROWCHUNK)
        {
            for(k = 0; k < size; k += COLCHUNK)
            {
                for (int ii = i; ii < min(i + ROWCHUNK, size); ii += 8)
                {
                    for (int kk = k; kk < min(k + COLCHUNK, size); kk++)
                    {
                        #pragma ivdep
                        #pragma vector aligned
	                    for (int jj = j; jj < min(j + ROWCHUNK, size); jj ++)
                        {
                            register DTYPE tmp = B[kk][jj];

                            C[ii][jj] += A[ii][kk] * tmp;
                            C[ii + 1][jj] += A[ii + 1][kk] * tmp;
                            C[ii + 2][jj] += A[ii + 2][kk] * tmp;
                            C[ii + 3][jj] += A[ii + 3][kk] * tmp;
                            C[ii + 4][jj] += A[ii + 4][kk] * tmp;
                            C[ii + 5][jj] += A[ii + 5][kk] * tmp;
                            C[ii + 6][jj] += A[ii + 6][kk] * tmp;
                            C[ii + 7][jj] += A[ii + 7][kk] * tmp;
                        }
                    }
                }
            }
        }
    }
}

// tiling
void
m_tiling(int size, DTYPE* A, DTYPE* B, DTYPE* C, int tile_size)
{
    #pragma omp parallel for default(none) shared(A, B, C, size, tile_size)
    for(int i = 0; i < size; i += tile_size)
        for(int k = 0; k < size; k += tile_size)
            for(int j = 0; j < size; j += tile_size)
                for(int ii = i; ii < min(i + tile_size, size); ++ii)
                    for(int kk = k; kk < min(k + tile_size, size); ++kk)
                        for(int jj = j; jj < min(j + tile_size, size); ++jj)
                            C[ii * size + jj] += A[ii * size + kk] * B[kk * size + jj];
}

// tiling 2d
void
m_tiling_2d(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size)
{
    register double e;

    #pragma omp parallel for default(none) shared(A, B, C, size, tile_size) private(e)
    for(int i = 0; i < size; i += tile_size)
        for(int k = 0; k < size; k += tile_size)
            for(int j = 0; j < size; j += tile_size)
                for(int ii = i; ii < min(i + tile_size, size); ++ii)
                    for(int kk = k; kk < min(k + tile_size, size); ++kk)
                    {
                        e = A[ii][kk];
                        for(int jj = j; jj < min(j + tile_size, size); ++jj)
                            C[ii][jj] += e * B[kk][jj];
                    }
}

// elemental function
void
m_elem_fun2(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size)
{
    // #pragma omp parallel for default(none) shared(A, B, C, size, tile_size)
    for (int i = 0; i < size; i += tile_size)
    {
        // printf ("I %d\n", i);
        for (int k = 0; k < size; k += tile_size)
        {
            // printf ("K %d\n", k);
            for (int j = 0; j < size; j += tile_size)
            {
                // printf ("J %d\n", j);

                for(int ii = i; ii < min(i + tile_size, size); ++ii)
                {
                    ALIGN_CODE DTYPE* r = C[ii];
                    ALIGN_CODE DTYPE* u = A[ii];

                    // printf ("A: %.2f\n", *u);

                    for(int kk = k; kk < min(k + tile_size, size); ++kk)
                    {
                        for(int jj = j; jj < min(j + tile_size, size); ++jj)
                        {
                            ALIGN_CODE DTYPE* v = B[kk];

                            // printf ("ii=%d, kk=%d, jj=%d\n", ii, kk, 0);

                            // for (int m = 0; m < tile_size; m++)
                            // {
                            //     printf ("%f %f\n", u[m], v[m]);
                            // }

                            // #pragma vector aligned
                            // 0:size or j
                            // mul_vect(&u[kk], &v[0:tile_size], &r[0:tile_size]);
                            mul_vect(&r[jj], &u[kk:tile_size], &v[jj:tile_size]);

                            // mul_vect(&r[kk], &u[kk:tile_size], &v[kk:tile_size]);
                            // printf ("\n");
                            // r[jj] += u[kk] * v[jj];
                        }
                    }
                }
            }


                        // C[ii * size + jj] += A[ii * size + kk] * B[kk * size + jj];



            // Cilk - useless
            // cilk_for (int j = 0; j < size; ++j) {
            // mul_vect(&u[k], &v[j], &r[j]);
            // }


            // mul_vect(&u[0:size], &v[0:size], &r[0:size]);
            // for (int j = 0; j < size; ++j)
                // r[j] += u[k] * v[j];

        }
    }
}

// ikj 1d notation with tiling
void
m_ikj_1d(int size, DTYPE* A, DTYPE* B, DTYPE* C, int tile_size)
{
    #pragma omp parallel for default(none) shared(A, B, C, size, tile_size)
    for (int i = 0; i < size; i += tile_size)
    {
        for (int k = 0; k < size; k += tile_size)
        {
            // #pragma vector aligned
            for (int j = 0; j < size; j += tile_size)
            {
                for(int ii = i; ii < min(i + tile_size, size); ++ii)
                {
                    DTYPE*  r = C + ii * size;
                    DTYPE*  u = A + ii * size;

                    for(int kk = k; kk < min(k + tile_size, size); ++kk)
                    {
                        DTYPE*  v = B + kk * size;

                        for(int jj = j; jj < min(j + tile_size, size); ++jj)
                        {
                            #pragma vector aligned
                            r[jj] += u[kk] * v[jj];
                        }
                    }
                }
            }
        }
    }
}

void
m_mkl(int size, DTYPE* A, DTYPE* B, DTYPE* C)
{
    DTYPE alpha = 1.0f, beta = 0.0f;

    MKL_FUNC(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, alpha,
                  A, size, B, size, beta, C, size);
    // sgemm("N", "N", &size, &size, &size, &alpha, (DTYPE*)A, &size, (DTYPE*)B, &size, &beta, (DTYPE*)C, &size);
}

void
m_choose (int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size)
{
    if (size < 2000)
        m_vect_2d (size, A, B, C);
    else
    {
        #if USE_ALIGNMENT
        DTYPE* A1 = MALLOC (sizeof (DTYPE) * size * size, ALIGN);
        DTYPE* B1 = MALLOC (sizeof (DTYPE) * size * size, ALIGN);
        DTYPE* C1 = MALLOC (sizeof (DTYPE) * size * size, ALIGN);

        #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            memmove(A1 + i * size, &A[i][0], size * sizeof(DTYPE));
            memmove(B1 + i * size, &B[i][0], size * sizeof(DTYPE));
            // memmove(C1 + i * size, &C[i][0], size * sizeof(DTYPE));
        }

        // print_matrix (size, A);
        // print_1d_matrix (size, A1);

        // print_matrix (size, B);
        // print_1d_matrix (size, B1);

        // print_matrix (size, C);
        // print_1d_matrix (size, C1);

        m_test(size, A1, B1, C1, tile_size);

        // print_matrix (size, C);

        m_vect_2d (size, A, B, C);

        // print_matrix (size, C);
        // print_1d_matrix (size, C1);

        #pragma omp parallel for
        for (int i = 0; i < size; i++)
            memmove(C[i], C1 + i * size, size * sizeof(DTYPE));

        FREE (A1);
        FREE (B1);
        FREE (C1);

        #endif
    }
}
