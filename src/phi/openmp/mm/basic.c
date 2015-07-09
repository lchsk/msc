#include "helper.h"
#include "basic.h"


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


// IKJ 2
void
m_ikj_2(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
    int i, j, k;
    #pragma omp parallel private(i,j,k)
    {
        #pragma omp for nowait
        for (i = 0; i < size; i++)
        {
            for (k = 0; k < size; k++)
            {
                for (j = 0; j < size; j++)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
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

void
m_ikj_unroll(int size, DTYPE** A, DTYPE** B, DTYPE** C)
{
    DTYPE temp;
    int k;
    #pragma omp parallel for default(none) shared(A, B, C, size, k) private(temp)
    for (int i=0;i<size;i++){
       for (int j=0;j<size;j++){
	   temp = 0.0;
    	   for (k=0;k<(size-3);k+=4){
	       temp += A[i][k]*B[k][j];
	       temp += A[i][k+1]*B[k+1][j];
	       temp += A[i][k+2]*B[k+2][j];
	       temp += A[i][k+3]*B[k+3][j];
	   }
	   for (;k<size;k++){
	       temp += A[i][k]*B[k][j];
	   }
	   C[i][j] = temp;
       }
    }

	// int uf = 4;
    // DTYPE* restrict tmp;
    //
    // // #pragma omp parallel for default(none) shared(A, B, C, size, uf) private(tmp)
	// for (int i = 0; i < size; i++)
    // {
	// 	for (int k = 0; k < size; k++)
    //     {
	// 		tmp = &A[i][k];
    //
	// 		for (int j = 0; j < size / uf; j += uf)
    //         {
	// 			C[i][j]     += *tmp * B[k][j];
	// 			C[i][j + 1] += *tmp * B[k][j + 1];
	// 			C[i][j + 2] += *tmp * B[k][j + 2];
	// 			C[i][j + 3] += *tmp * B[k][j + 3];
	// 		}
    //
	// 		for (int j = size - uf; j < size; j++)
	// 			C[i][j] += *tmp * B[k][j];
	// 	}
	// }
}
