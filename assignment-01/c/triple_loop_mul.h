#ifndef TRIPLE_LOOP_MUL_H
#define TRIPLE_LOOP_MUL_H

#include <stddef.h>

/*
 * out = a * b for three row-major n x n double matrices.
 *
 * Precondition: a, b and out each point to n*n doubles and out does not
 * alias a or b. O(n^3) arithmetic work; every entry of out is overwritten.
 */
void triple_loop_mul_ijk(const double *a, const double *b, double *out, size_t n);

#endif
