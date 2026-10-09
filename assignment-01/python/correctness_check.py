"""Reference-based correctness checking for the Python kernel.

Writes the small shared test resources (A, B and the NumPy float64 reference
product for n = 5) into data/reference/ and verifies that the pure-Python
triple loop matches the reference within rtol=1e-9, atol=1e-12. The Java, C
and Go test suites consume the same three .bin files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from generate_inputs import SEED, generate_pair, write_matrix
from triple_loop_mul import multiply_into, zero_matrix

REFERENCE_SIZE = 5
RTOL = 1e-9
ATOL = 1e-12
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"


def numpy_reference(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    return matrix_a @ matrix_b


def check_against_reference(python_product: np.ndarray, reference: np.ndarray) -> None:
    if not np.allclose(python_product, reference, rtol=RTOL, atol=ATOL):
        worst = np.max(np.abs(python_product - reference))
        raise AssertionError(f"python kernel deviates from NumPy reference, max abs error {worst}")


def write_reference_bundle(size: int, reference_dir: Path, seed: int = SEED) -> None:
    matrix_a, matrix_b = generate_pair(size, seed)
    reference = numpy_reference(matrix_a, matrix_b)
    reference_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(reference_dir / f"matrixA_{size}.bin", matrix_a)
    write_matrix(reference_dir / f"matrixB_{size}.bin", matrix_b)
    write_matrix(reference_dir / f"expectedC_{size}.bin", reference)


def sequential_checksum(matrix: list[list[float]]) -> float:
    """Row-major left-to-right sum; matches the runners' checksum order."""
    return sum(sum(row) for row in matrix)


def verify_python_kernel(size: int, reference_dir: Path, seed: int = SEED) -> float:
    matrix_a, matrix_b = generate_pair(size, seed)
    out = zero_matrix(size)
    multiply_into(matrix_a.tolist(), matrix_b.tolist(), out)
    python_product = np.array(out, dtype=np.float64)
    reference = numpy_reference(matrix_a, matrix_b)
    check_against_reference(python_product, reference)
    return sequential_checksum(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--size", type=int, default=REFERENCE_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_reference_bundle(args.size, args.reference_dir)
    checksum = verify_python_kernel(args.size, args.reference_dir)
    print(f"python kernel matches NumPy reference (rtol={RTOL}, atol={ATOL})")
    print(f"checksum n={args.size}: {checksum:.17g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
