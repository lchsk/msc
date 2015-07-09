#ifndef FAST_H
#define FAST_H

// m_vect_2d
void m_vect_2d(int size, DTYPE** A, DTYPE** B, DTYPE** C);

// m_vect_2d_tiled
void m_vect_2d_tiled(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size);

// array notation
void m_array_not(int size, DTYPE* A, DTYPE* B, DTYPE* C);

// elemental function
void m_elem_fun(int size, DTYPE** A, DTYPE** B, DTYPE** C, int tile_size);

#endif
