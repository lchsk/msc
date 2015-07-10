#include "helper.h"
#include "fast.h"
#include "elemental.h"

// m_vect_2d
void
m_vect_2d(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
    #pragma omp parallel for shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        ALIGN_CODE DTYPE* restrict r = C[i];
        ALIGN_CODE DTYPE* restrict u = A[i];

        for (int k = 0; k < size; ++k)
        {
            ALIGN_CODE DTYPE* restrict v = B[k];

            #pragma vector aligned
            // #pragma unroll(8)
            // #pragma ivdep
            for (int j = 0; j < size; ++j)
            {
                #pragma vector aligned
                r[j] += u[k] * v[j];
            }
        }
    }
}

// m_vect_2d_tiled
void
m_vect_2d_tiled(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size)
{
    #pragma omp parallel for collapse(2) shared(A, B, C, size, tile_size)
    for (int i = 0; i < size; i += tile_size)
    {
        // printf ("I %d\n", i);
        for (int k = 0; k < size; k += tile_size)
        {
            // printf ("K %d\n", k);
            for (int j = 0; j < size; j += tile_size)
            {
                // printf ("J %d\n", j);

                for(int ii = i; ii < min(i + tile_size, size); ii++)
                {
                    ALIGN_CODE DTYPE* r = C[ii];
                    ALIGN_CODE DTYPE* u = A[ii];

                    // printf ("A: %.2f\n", *u);

                    for(int kk = k; kk < min(k + tile_size, size); ++kk)
                    {
                        #pragma vector aligned
                        #pragma ivdep
                        for(int jj = j; jj < min(j + tile_size, size); ++jj)
                        {
                            ALIGN_CODE DTYPE* v = B[kk];
                            #pragma vector aligned
                            r[jj] += u[kk] * v[jj];
                        }
                    }
                }
            }



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
        {
            #pragma vector aligned
            C[i * size:size] += A[i * size + k] * B[k * size:size];
        }
    }
}

// elemental function
void
m_elem_fun(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size)
{
    #pragma omp parallel for default(none) shared(A, B, C, size)
    for (int i = 0; i < size; ++i)
    {
        DTYPE* r = C[i];
        DTYPE* u = A[i];

        #pragma vector aligned
        for (int k = 0; k < size; ++k)
        {
            DTYPE* v = B[k];
            mul_vect(&r[0:size], &v[0:size], &u[k]);

        }
    }
}
