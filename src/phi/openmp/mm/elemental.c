#include "helper.h"
#include "elemental.h"

__attribute__((vector))void mul_vect(DTYPE* c, DTYPE* a, DTYPE* b)
{
    // c[0] = a[0] * b[0];

    *c += *a * *b;
    // printf ("%.2f x %.2f = %.2f\n", *a, *b, *c);
    // return;
}
