#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// #include <mkl.h>
#include <time.h>
// #include <cilk/cilk.h>
#include "helper.h"
#include "basic.h"
#include "experimental.h"
#include "elemental.h"
#include "fast.h"

// Available implementations

#define IJK 1
#define IKJ_2 0
#define IJK_RESTRICT 0
#define IKJ 0
#define IKJ_RESTRICT 0
#define IKJ_RESTRICT_TMP 0
#define IKJ_UNROLL 0

#define IKJ_VECT_2D 0
#define IKJ_VECT_2D_TILED 0
#define IKJ_1D_NOTATION 0

#define ARRAY_NOTATION 0

#define TILING 0
#define TILING_2D 0
#define TEST 0

#define ELEM_FUNCTION 0

void init_matrices_2d (int size, DTYPE*** A, DTYPE*** B, DTYPE*** C)
{
    *A = new_matrix(size);
    *B = new_matrix(size);
    *C = new_matrix(size);
    init_matrix(size, *A);
    init_matrix(size, *B);
}


int main(int argc, char *argv[])
{
    srand(time(NULL));

    int size = 1000;
    int iter = 1;
    int tile_size = 8;

    if (argc >= 2 && argv[1] != NULL)
        size = atoi (argv[1]);

    if (argc >= 3 && argv[2] != NULL)
        iter = atoi (argv[2]);

    if (argc >= 4 && argv[3] != NULL)
        tile_size = atoi (argv[3]);

    int threads = 4;

    double t_avg = 0.0;
    double t_start;

    ALIGN_CODE DTYPE** A = NULL;
    ALIGN_CODE DTYPE** B = NULL;
    ALIGN_CODE DTYPE** C = NULL;

    ALIGN_CODE DTYPE* A1 = NULL;
    ALIGN_CODE DTYPE* B1 = NULL;
    ALIGN_CODE DTYPE* C1 = NULL;

    #pragma omp parallel
    #pragma omp master
    printf ("Size: %d x %d Iterations: %d Threads: %d Alignment: %d MIC: %d Dtype: %db Tile: %d\
            \n\n",
        size, size, iter, omp_get_num_threads(), USE_ALIGNMENT, MIC, sizeof (DTYPE) * 8, tile_size
    );
    #pragma omp barrier

    // ----------------------------------------------





    // A = new_matrix(size);
    // B = new_matrix(size);
    // C = new_matrix(size);
    // init_matrix(size, A);
    // init_matrix(size, B);

    #if IJK
        init_matrices_2d (size, &A, &B, &C);

        t_start = omp_get_wtime();

        for (int idx = 0; idx < iter; idx++)
            m_ijk(size, A, B, C);

        print_results ("IJK", t_start, size, iter, 2, 1, A, B, C, A1, B1, C1);
        // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
        free_matrices (size, A, B, C);
    #endif

    // ----------------------------------------------

    // IKJ 2

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
    //     m_ikj_2(size, A, B, C);
    // }
    //
    // print_results ("IKJ 2", t_start, size, iter);
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

    // A = new_matrix(size);
    //
    // B = new_matrix(size);
    // C = new_matrix(size);
    // init_matrix(size, A);
    // init_matrix(size, B);
    //
    // t_start = omp_get_wtime();
    //
    // for (int idx = 0; idx < iter; idx++)
    // {
    //     m_ikj(size, A, B, C);
    // }
    //
    // print_results ("IKJ", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ restrict

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
    //     m_ikj_restrict(size, A, B, C);
    // }
    //
    // print_results ("IKJ restrict", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ restrict tmp

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
    //     m_ikj_restrict_tmp(size, A, B, C);
    // }
    //
    // print_results ("IKJ restrict tmp", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ vect 2d

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
    //     m_vect_2d(size, A, B, C);
    // }
    //
    //
    // print_results ("IKJ vect 2d", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
    // // print_matrix (size, A);
    // // print_matrix (size, B);
    // // print_matrix (size, C);
    //
    //
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ vect 2d tiled

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
    //     m_vect_2d_tiled(size, A, B, C, tile_size);
    // }
    //
    //
    // print_results ("IKJ vect 2d tiled", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
    // // print_matrix (size, A);
    // // print_matrix (size, B);
    // // print_matrix (size, C);
    //
    //
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // IKJ 1d notation

    // A1 = new_1d_matrix(size);
    // B1 = new_1d_matrix(size);
    // C1 = new_1d_matrix(size);
    // init_1d_matrix(size, A1);
    // init_1d_matrix(size, B1);
    //
    // t_start = omp_get_wtime();
    //
    // for (int idx = 0; idx < iter; idx++)
    // {
    //     m_ikj_1d(size, A1, B1, C1, tile_size);
    // }
    //
    // print_results ("IKJ 1d notation", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_1d (size, A1, B1, C1));
    // free_1d_matrices (A1, B1, C1);

    // ----------------------------------------------

    // array notation

    // A1 = new_1d_matrix(size);
    // B1 = new_1d_matrix(size);
    // C1 = new_1d_matrix(size);
    // init_1d_matrix(size, A1);
    // init_1d_matrix(size, B1);
    //
    // t_start = omp_get_wtime();
    //
    // for (int idx = 0; idx < iter; idx++)
    // {
    //     m_array_not(size, A1, B1, C1);
    // }
    //
    // print_results ("Array notation", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_1d (size, A1, B1, C1));
    // free_1d_matrices (A1, B1, C1);

    // ----------------------------------------------

    // IKJ unroll

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
    //     m_ikj_unroll (size, A, B, C);
    // }
    //
    //
    // print_results ("IKJ unroll", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
    // // print_matrix (size, A);
    // // print_matrix (size, B);
    // // print_matrix (size, C);
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // tiling

    // A1 = new_1d_matrix(size);
    // B1 = new_1d_matrix(size);
    // C1 = new_1d_matrix(size);
    // init_1d_matrix(size, A1);
    // init_1d_matrix(size, B1);
    //
    // t_start = omp_get_wtime();
    //
    // for (int idx = 0; idx < iter; idx++)
    // {
    //     m_tiling(size, A1, B1, C1, tile_size);
    // }
    //
    // print_results ("Tiling", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_1d (size, A1, B1, C1));
    // free_1d_matrices (A1, B1, C1);

    // ----------------------------------------------

    // test

    // A1 = new_1d_matrix(size);
    // B1 = new_1d_matrix(size);
    // C1 = new_1d_matrix(size);
    // init_1d_matrix(size, A1);
    // init_1d_matrix(size, B1);
    //
    // t_start = omp_get_wtime();
    //
    // for (int idx = 0; idx < iter; idx++)
    // {
    //     m_test(size, A1, B1, C1, tile_size);
    // }
    //
    // print_results ("Test", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_1d (size, A1, B1, C1));
    // free_1d_matrices (A1, B1, C1);

    // ----------------------------------------------

    // tiling 2d

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
    //     m_tiling_2d(size, A, B, C, tile_size);
    // }
    //
    // print_results ("Tiling 2d", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
    // free_matrices (size, A, B, C);

    // ----------------------------------------------

    // Elem function

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
    //     m_elem_fun(size, A, B, C, tile_size);
    // }
    //
    // print_results ("Elemental function", t_start, size, iter);
    // printf ("\tCorrect: %d\n", is_correct_2d (size, A, B, C));
    //
    //
    // free_matrices (size, A, B, C);

    // DTYPE a[] = {1,2,3};
    // DTYPE b[] = {1,2,3};
    // DTYPE c[] = {0,0,0};

    // mul_vect(&a[:], &b[:], &c[:]);
    //
    // for (int i = 0; i < 3; i++)
    //     printf ("%f ", c[i]);

    return 0;
}
