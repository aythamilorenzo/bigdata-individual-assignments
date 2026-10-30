"""Generate the shared dense input matrices reused from assignments 1 and 2.

Each n in the campaign size sequence produces two files, matrixA_n.bin and
matrixB_n.bin, containing float64 values in little-endian row-major order.
These are the exact inputs validated and benchmarked in assignments 1 and 2
for the shared sizes; assignment 3 reuses them unchanged so the sequential
baseline and the parallel method multiply the same logical matrices as the
earlier studies. The generator is deterministic: each size n uses NumPy's
PCG64 seeded with the integer sequence [42, n] (so any single size can be
regenerated in isolation), drawing A first and then B from the same stream,
with values drawn uniformly from [0, 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SEED = 42
SIZES: tuple[int, ...] = (500, 1000, 1500, 2000, 3000, 4000)
VALUE_RANGE = "uniform [0, 1)"
NUMERIC_TYPE = "float64 little-endian, row-major"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "inputs"


def generate_pair(size: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return two size x size float64 matrices drawn from one seeded stream."""
    rng = np.random.default_rng([seed, size])
    matrix_a = rng.uniform(0.0, 1.0, size=(size, size))
    matrix_b = rng.uniform(0.0, 1.0, size=(size, size))
    return matrix_a, matrix_b


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    """Write one matrix as little-endian float64 row-major bytes."""
    path.write_bytes(np.ascontiguousarray(matrix, dtype="<f8").tobytes())


def generate_inputs(sizes: tuple[int, ...], out_dir: Path, seed: int = SEED) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        matrix_a, matrix_b = generate_pair(size, seed)
        write_matrix(out_dir / f"matrixA_{size}.bin", matrix_a)
        write_matrix(out_dir / f"matrixB_{size}.bin", matrix_b)
        print(f"generated n={size}: {2 * size * size * 8} bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for the .bin files (default: data/inputs)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SIZES),
        help="side lengths to generate (default: the campaign sequence)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="base RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_inputs(tuple(args.sizes), args.out_dir, args.seed)


if __name__ == "__main__":
    main()
