#ifndef BASIC_H
#define BASIC_H 1

// IKJ version
void m_ikj(int size, DTYPE** A, DTYPE** B, DTYPE** C);

// IKJ 2
void m_ikj_2(int size, DTYPE** A, DTYPE** B, DTYPE** C);

// IKJ + restrict version
void m_ikj_restrict(int size, DTYPE** restrict A, DTYPE** restrict B, DTYPE** restrict C);

// IKJ + restrict + tmp version
void m_ikj_restrict_tmp(int size, DTYPE** restrict A, DTYPE** restrict B, DTYPE** restrict C);

// IJK + restrict version
void m_ijk_restrict(int size, DTYPE** restrict A, DTYPE** restrict B, DTYPE** restrict C);

void m_ikj_unroll(int size, DTYPE** A, DTYPE** B, DTYPE** C);

// IJK version
void m_ijk(int size, DTYPE** A, DTYPE** B, DTYPE** C);

#endif
