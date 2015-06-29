#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

float f(float x)
{
    return x * x;
}

int main(int argc, char* argv[])
{
    int nthreads;

    if (argv[1] == NULL)
        nthreads = omp_get_max_threads();
    else
        nthreads = atoi(argv[1]);

    printf("Using %d threads\n", nthreads);

    omp_set_num_threads(nthreads);

    double t;
    float r;

    int iter = 100000;

    t = omp_get_wtime();

    // #pragma omp simd collapse(2)
    #pragma omp parallel for
    for (int j = 0; j < iter; j++)
    {
        float a = 0.0;
        float b = 5919595.0;
        float steps = 1000.0;
        float step = (b - a) / steps;

        float m = b - a;

        r = 0.0;

        // for (float i = a; i < b; i += step)
        for (int i = 0; i < steps; i++)
        {
            // r += step * ((f(i) + f(i + step)) / 2.0);
            r += step * ((f(a + i * step) + f(a + i * step + step)) / 2.0);
        }
    }

    printf("Time: %f ms\n", (omp_get_wtime() - t) * 1000);

    printf("Result: %f\n", r);

    return 0;
}