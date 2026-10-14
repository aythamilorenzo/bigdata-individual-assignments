"""Generate the declared-density sparse input pairs for the Part B campaign.

For every (size, density) configuration this script writes two dense-layout
little-endian float64 files, matrixA_n{size}_d{density}.bin and
matrixB_n{size}_d{density}.bin, containing exactly round(density/100 * n^2)
nonzero entries drawn uniformly without replacement over the n^2 positions;
their values are uniform in [0, 1), the same value range as the dense inputs
of assignment 1. Density is the percentage of nonzero entries, sparsity the
complementary percentage of zero entries.

The same files feed both representations of the matched comparison: the CSR
kernel converts them to sparse form, the dense baseline multiplies them as
loaded, so both methods see the identical logical matrices.

The generator is deterministic: each (size, density) configuration uses
NumPy's PCG64 seeded with [42, size, density_percent], drawing A first and
then B from the same stream. The realised nonzero counts are recorded in
data/sparse/inputs_manifest.csv, because a uniformly drawn position set may
deviate slightly from the nominal count only through the rounding of the
target (the sampler always draws exactly the requested number of positions).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SEED = 42
SIZES: tuple[int, ...] = (100, 500, 1000, 2000)
DENSITIES_PERCENT: tuple[int, ...] = (1, 5, 10, 20, 50)
VALUE_RANGE = "uniform [0, 1)"
NUMERIC_TYPE = "float64 little-endian, row-major"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "sparse"
MANIFEST_NAME = "inputs_manifest.csv"
MANIFEST_HEADER = [
    "size", "density_percent", "sparsity_percent", "nnz_a", "nnz_b",
    "seed_sequence", "value_range", "numeric_type", "position_distribution",
]


def generate_sparse_pair(size: int, density_percent: int, seed: int = SEED
                         ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return two size x size matrices with exactly the target nonzero count."""
    rng = np.random.default_rng([seed, size, density_percent])
    target_nnz = round(density_percent / 100 * size * size)
    matrix_a = build_matrix(size, target_nnz, rng)
    matrix_b = build_matrix(size, target_nnz, rng)
    return matrix_a, matrix_b, count_nonzeros(matrix_a), count_nonzeros(matrix_b)


def build_matrix(size: int, target_nnz: int, rng: np.random.Generator) -> np.ndarray:
    """Draw target_nnz distinct positions uniformly and fill them with values."""
    flat = np.zeros(size * size, dtype="<f8")
    positions = rng.choice(size * size, size=target_nnz, replace=False)
    flat[positions] = rng.uniform(0.0, 1.0, size=target_nnz)
    return flat.reshape(size, size)


def count_nonzeros(matrix: np.ndarray) -> int:
    return int(np.count_nonzero(matrix))


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    """Write one matrix as little-endian float64 row-major bytes."""
    path.write_bytes(np.ascontiguousarray(matrix, dtype="<f8").tobytes())


def input_stem(size: int, density_percent: int) -> str:
    return f"n{size}_d{density_percent}"


def generate_inputs(sizes: tuple[int, ...], densities: tuple[int, ...], out_dir: Path,
                    seed: int = SEED) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    rows: list[list[object]] = []
    for size in sizes:
        for density_percent in densities:
            matrix_a, matrix_b, nnz_a, nnz_b = generate_sparse_pair(
                size, density_percent, seed)
            stem = input_stem(size, density_percent)
            write_matrix(out_dir / f"matrixA_{stem}.bin", matrix_a)
            write_matrix(out_dir / f"matrixB_{stem}.bin", matrix_b)
            rows.append([
                size, density_percent, 100 - density_percent, nnz_a, nnz_b,
                f"[{seed}, {size}, {density_percent}]", VALUE_RANGE,
                NUMERIC_TYPE, "uniform without replacement",
            ])
            print(f"generated n={size} density={density_percent}%: "
                  f"nnz_a={nnz_a} nnz_b={nnz_b}")
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(rows)
    print(f"manifest written to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for the .bin files (default: data/sparse)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SIZES),
        help="side lengths to generate (default: the campaign sequence)",
    )
    parser.add_argument(
        "--densities",
        type=int,
        nargs="+",
        default=list(DENSITIES_PERCENT),
        help="nonzero percentages to generate (default: the campaign sequence)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="base RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_inputs(tuple(args.sizes), tuple(args.densities), args.out_dir, args.seed)


if __name__ == "__main__":
    main()
