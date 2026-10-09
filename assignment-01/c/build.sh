#!/usr/bin/env bash
# Build the C kernel, benchmark runner and correctness test.
# Fixed flags for the whole campaign (document any change in the README).
set -euo pipefail
cd "$(dirname "$0")"

CFLAGS="-O2 -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L"

gcc $CFLAGS -o matrixmul_benchmark triple_loop_mul.c benchmark_main.c
gcc $CFLAGS -o correctness_test triple_loop_mul.c correctness_test.c -lm

echo "C build OK (no warnings): matrixmul_benchmark, correctness_test"
