#ifndef EXPERIMENTAL_H
#define EXPERIMENTAL_H

void m_test(int size, DTYPE* A, DTYPE* B, DTYPE* C, int tile_size);

// tiling
void m_tiling(int size, DTYPE* A, DTYPE* B, DTYPE* C, int tile_size);

// tiling 2d
void m_tiling_2d(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size);

// elemental function
void m_elem_fun2(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size);

// ikj 1d notation with tiling
void m_ikj_1d(int size, DTYPE* A, DTYPE* B, DTYPE* C, int tile_size);

void m_mkl(int size, DTYPE* A, DTYPE* B, DTYPE* C);

#endif
