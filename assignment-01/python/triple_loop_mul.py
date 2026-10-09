"""Basic dense matrix multiplication: the straightforward i-j-k triple loop.

Matrices are lists of rows of Python floats (IEEE-754 binary64), which matches
row-major storage in the compiled languages. The kernel performs O(n^3)
arithmetic work and writes every entry of ``out``; it must stay free of any
library call (no NumPy) because this is the code path measured by the
benchmark.
"""

from __future__ import annotations


def multiply_into(matrix_a: list[list[float]],
                  matrix_b: list[list[float]],
                  out: list[list[float]]) -> None:
    """Compute out = matrix_a @ matrix_b with the i-j-k triple loop."""
    n = len(matrix_a)
    for i in range(n):
        row_a = matrix_a[i]
        row_out = out[i]
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += row_a[k] * matrix_b[k][j]
            row_out[j] = total


def zero_matrix(n: int) -> list[list[float]]:
    """Return an n x n matrix filled with 0.0."""
    return [[0.0] * n for _ in range(n)]
