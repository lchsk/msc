#ifndef HELPER_H
#define HELPER_H

// Aliases

#define DTYPE double

// How a matrix is represented
#define REPR_2D 2
#define REPR_1D 1

#define TRUE 1
#define FALSE 0

// Options

#define USE_ALIGNMENT 1

#ifdef __MIC__
#define MIC 1
#define VEC_LEN 64
#else
#define MIC 0
#define VEC_LEN 64
#endif

#ifdef USE_ALIGNMENT
#define MALLOC _mm_malloc
#define FREE _mm_free
#define ALIGN VEC_LEN
#define ALIGN_CODE __attribute__((aligned(ALIGN)))
#else
#define MALLOC malloc
#define FREE free
#define ALIGN
#define ALIGN_CODE
#endif

// Macros

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))


ALIGN_CODE DTYPE**
new_matrix(int size);

ALIGN_CODE DTYPE*
new_1d_matrix(int size);

void
free_matrices(int size, DTYPE** A, DTYPE** B, DTYPE** C);

void
free_1d_matrices (DTYPE* A, DTYPE* B, DTYPE* C);

void
init_matrix(int size, DTYPE** m);

void
init_1d_matrix(int size, DTYPE* m);

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
);

void
print_matrix (int size, DTYPE** m);

void
print_1d_matrix (int size, DTYPE* m);

int
is_correct_2d (int size, DTYPE** A, DTYPE** B, DTYPE** C);

int
is_correct_1d (int size, DTYPE* A, DTYPE* B, DTYPE* C);

void
init_matrices_2d (int size, DTYPE*** A, DTYPE*** B, DTYPE*** C, double* time);

void
init_matrices_1d (int size, DTYPE** A, DTYPE** B, DTYPE** C, double* time);

#endif
