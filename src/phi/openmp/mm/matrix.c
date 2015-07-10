#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <time.h>
// #include <cilk/cilk.h>
#include "helper.h"
#include "basic.h"
#include "experimental.h"
#include "elemental.h"
#include "fast.h"

#if USE_MKL
#include <mkl.h>
#endif

// Available implementations

#define IJK FALSE
#define IKJ_2 FALSE
#define IJK_RESTRICT FALSE
#define IKJ FALSE
#define IKJ_RESTRICT FALSE
#define IKJ_RESTRICT_TMP FALSE
#define IKJ_UNROLL FALSE

#define IKJ_VECT_2D TRUE
#define IKJ_VECT_2D_TILED FALSE
#define IKJ_1D_NOTATION FALSE

#define ARRAY_NOTATION FALSE

#define TILING FALSE
#define TILING_2D FALSE
#define TEST FALSE

#define ELEM_FUNCTION FALSE

#define MKL TRUE

// ---


int main(int argc, char *argv[])
{
    srand(time(NULL));

    int size = 1000;
    int iter = 1;
    int tile_size = 8;
    int mic_count = 0;
    int auto_offload = -1; // not set

    // if (getenv ("MKL_MIC_ENABLE") != NULL)
        // auto_offload = atoi (getenv ("MKL_MIC_ENABLE"));

    #if ( ! MIC && USE_MKL && ENABLE_AUTO_OFFLOAD)
        auto_offload = TRUE;
        mkl_mic_enable();
    #elif ( ! MIC)
        auto_offload = FALSE;
        mkl_mic_disable();
    #endif

    #if (USE_MKL && ! MIC)
        mic_count = mkl_mic_get_device_count();
    #endif

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
    printf ("Size: %d x %d Iterations: %d Threads: %d Alignment: %d MIC: %d Dtype: %db Tile: %d MIC units: %d Warm up: %d AO: %d\
            \n\n",
        size, size, iter, omp_get_num_threads(), USE_ALIGNMENT, MIC, sizeof (DTYPE) * 8, tile_size, mic_count, CHEAT, auto_offload
    );
    #pragma omp barrier

    // ----------------------------------------------

    #if IJK
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_ijk(size, A, B, C);

        print_results ("IJK", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if IKJ_2
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_ikj_2(size, A, B, C);

        print_results ("IKJ_2", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if IJK_RESTRICT
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_ijk_restrict(size, A, B, C);

        print_results ("IJK_RESTRICT", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if IKJ
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_ikj(size, A, B, C);

        print_results ("IKJ", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if IKJ_RESTRICT
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_ikj_restrict(size, A, B, C);

        print_results ("IKJ_RESTRICT", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if IKJ_RESTRICT_TMP
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_ikj_restrict_tmp(size, A, B, C);

        print_results ("IKJ_RESTRICT_TMP", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if CHEAT
        init_matrices_2d (size, &A, &B, &C, &t_start);
        m_vect_2d(size, A, B, C);
    #endif

    #if IKJ_VECT_2D
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_vect_2d(size, A, B, C);

        print_results ("IKJ_VECT_2D", t_start, size, iter, REPR_2D, CHECK_CORRECTNESS, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if IKJ_VECT_2D_TILED
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_vect_2d_tiled(size, A, B, C, tile_size);

        print_results ("IKJ_VECT_2D_TILED", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if IKJ_1D_NOTATION
        init_matrices_1d (size, &A1, &B1, &C1, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_ikj_1d(size, A1, B1, C1, tile_size);

        print_results ("IKJ_1D_NOTATION", t_start, size, iter, REPR_1D, TRUE, A, B, C, A1, B1, C1);
        free_1d_matrices (A1, B1, C1);
    #endif

    #if ARRAY_NOTATION
        init_matrices_1d (size, &A1, &B1, &C1, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_array_not(size, A1, B1, C1);

        print_results ("ARRAY_NOTATION", t_start, size, iter, REPR_1D, TRUE, A, B, C, A1, B1, C1);
        free_1d_matrices (A1, B1, C1);
    #endif

    #if IKJ_UNROLL
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_ikj_unroll(size, A, B, C);

        print_results ("IKJ_UNROLL", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if TILING
        init_matrices_1d (size, &A1, &B1, &C1, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_tiling(size, A1, B1, C1, tile_size);

        print_results ("TILING", t_start, size, iter, REPR_1D, TRUE, A, B, C, A1, B1, C1);
        free_1d_matrices (A1, B1, C1);
    #endif

    #if TEST
        init_matrices_1d (size, &A1, &B1, &C1, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_test(size, A1, B1, C1, tile_size);

        print_results ("TEST", t_start, size, iter, REPR_1D, TRUE, A, B, C, A1, B1, C1);
        free_1d_matrices (A1, B1, C1);
    #endif

    #if TILING_2D
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_tiling_2d(size, A, B, C, tile_size);

        print_results ("TILING_2D", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if ELEM_FUNCTION
        init_matrices_2d (size, &A, &B, &C, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_elem_fun(size, A, B, C, tile_size);

        print_results ("ELEM_FUNCTION", t_start, size, iter, REPR_2D, TRUE, A, B, C, A1, B1, C1);
        free_matrices (size, A, B, C);
    #endif

    #if CHEAT
        init_matrices_1d (size, &A1, &B1, &C1, &t_start);
        m_mkl(size, A1, B1, C1);
    #endif

    #if MKL
        init_matrices_1d (size, &A1, &B1, &C1, &t_start);

        for (int idx = 0; idx < iter; idx++)
            m_mkl(size, A1, B1, C1);

        print_results ("MKL", t_start, size, iter, REPR_1D, CHECK_CORRECTNESS, A, B, C, A1, B1, C1);
        free_1d_matrices (A1, B1, C1);
    #endif

    // DTYPE a[] = {1,2,3};
    // DTYPE b[] = {1,2,3};
    // DTYPE c[] = {0,0,0};

    // mul_vect(&a[:], &b[:], &c[:]);
    //
    // for (int i = 0; i < 3; i++)
    //     printf ("%f ", c[i]);

    return 0;
}
