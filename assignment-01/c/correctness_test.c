#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "triple_loop_mul.h"

#define REFERENCE_SIZE 5
#define RTOL 1e-9
#define ATOL 1e-12

static int failures = 0;

static void expect_true(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static double *load_matrix(const char *path, int size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    size_t elements = (size_t)size * (size_t)size;
    double *matrix = (double *)malloc(elements * sizeof(double));
    if (matrix == NULL) {
        fclose(file);
        return NULL;
    }
    if (fread(matrix, sizeof(double), elements, file) != elements) {
        fprintf(stderr, "%s: short read\n", path);
        free(matrix);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return matrix;
}

static void check_identity(void) {
    enum { n = 4 };
    double matrix_a[n * n];
    double identity[n * n];
    double out[n * n];
    unsigned int state = 7u;
    for (int index = 0; index < n * n; index++) {
        state = state * 1664525u + 1013904223u;
        matrix_a[index] = (double)(state % 1000u) / 1000.0;
        out[index] = 0.0;
    }
    memset(identity, 0, sizeof(identity));
    for (int i = 0; i < n; i++) {
        identity[i * n + i] = 1.0;
    }

    triple_loop_mul_ijk(matrix_a, identity, out, n);
    expect_true(memcmp(matrix_a, out, sizeof(out)) == 0,
                "A * I must equal A exactly");

    triple_loop_mul_ijk(identity, matrix_a, out, n);
    expect_true(memcmp(matrix_a, out, sizeof(out)) == 0,
                "I * A must equal A exactly");
}

static void check_zero(void) {
    enum { n = 4 };
    double matrix_a[n * n];
    double zero[n * n];
    double out[n * n];
    for (int index = 0; index < n * n; index++) {
        matrix_a[index] = (double)index;
        zero[index] = 0.0;
        out[index] = 1.0;
    }

    triple_loop_mul_ijk(matrix_a, zero, out, n);
    for (int index = 0; index < n * n; index++) {
        expect_true(out[index] == 0.0, "A * 0 must be 0");
    }
}

static void check_known_answer(void) {
    double matrix_a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double matrix_b[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    double expected[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};
    double out[9] = {0};

    triple_loop_mul_ijk(matrix_a, matrix_b, out, 3);
    for (int index = 0; index < 9; index++) {
        expect_true(out[index] == expected[index], "3x3 known-answer product");
    }
}

static int check_numpy_reference(const char *reference_dir) {
    char path_a[1024];
    char path_b[1024];
    char path_c[1024];
    snprintf(path_a, sizeof(path_a), "%s/matrixA_%d.bin", reference_dir, REFERENCE_SIZE);
    snprintf(path_b, sizeof(path_b), "%s/matrixB_%d.bin", reference_dir, REFERENCE_SIZE);
    snprintf(path_c, sizeof(path_c), "%s/expectedC_%d.bin", reference_dir, REFERENCE_SIZE);

    double *matrix_a = load_matrix(path_a, REFERENCE_SIZE);
    double *matrix_b = load_matrix(path_b, REFERENCE_SIZE);
    double *expected = load_matrix(path_c, REFERENCE_SIZE);
    if (matrix_a == NULL || matrix_b == NULL || expected == NULL) {
        free(matrix_a);
        free(matrix_b);
        free(expected);
        return -1;
    }
    size_t elements = (size_t)REFERENCE_SIZE * REFERENCE_SIZE;
    double *out = (double *)calloc(elements, sizeof(double));

    triple_loop_mul_ijk(matrix_a, matrix_b, out, REFERENCE_SIZE);
    for (size_t index = 0; index < elements; index++) {
        double tolerance = ATOL + RTOL * fabs(expected[index]);
        expect_true(fabs(out[index] - expected[index]) <= tolerance,
                    "n=5 entry within tolerance of the NumPy reference");
    }

    free(matrix_a);
    free(matrix_b);
    free(expected);
    free(out);
    return 0;
}

int main(int argc, char **argv) {
    const char *reference_dir = argc > 1 ? argv[1] : "data/reference";
    check_identity();
    check_zero();
    check_known_answer();
    if (check_numpy_reference(reference_dir) != 0) {
        fprintf(stderr, "FAIL: reference files missing in %s\n", reference_dir);
        return 1;
    }
    if (failures > 0) {
        fprintf(stderr, "%d correctness check(s) failed\n", failures);
        return 1;
    }
    printf("all C correctness checks passed (rtol=%g, atol=%g)\n", RTOL, ATOL);
    return 0;
}
