#ifndef HELPER_H
#define HELPER_H

#define DTYPE double

DTYPE**
new_matrix(int size);

DTYPE*
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
print_results (char* str, double t_start, int size, int iter);

void
print_matrix (int size, DTYPE** m);

void
print_1d_matrix (int size, DTYPE* m);

int
is_correct_2d (int size, DTYPE** A, DTYPE** B, DTYPE**C);

#endif
