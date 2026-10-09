#include "triple_loop_mul.h"

void triple_loop_mul_ijk(const double *a, const double *b, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double total = 0.0;
            for (size_t k = 0; k < n; k++) {
                total += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = total;
        }
    }
}
