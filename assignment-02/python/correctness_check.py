"""Cross-check the compiled jar against a NumPy reference on known matrices.

The JUnit suite already validates every kernel against the dense baseline in
JVM. This script adds an independent, out-of-JVM check: it runs the shaded
jar on the 5x5 reference matrices of assignment 1 (and on a small
declared-density pair generated for the sparse path) and compares the
checksums printed by the runner with NumPy float64 results.

Tolerances: rtol=1e-9, atol=1e-12 — the same tolerances as assignment 1 —
because the kernels sum in different orders (i-j-k vs i-k-j vs accumulator
sweep) and floating-point addition is not associative.

    correctness_check.py [--jar PATH]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DENSE_METHODS = ("triple_loop_ijk_dense", "triple_loop_ikj_dense", "blocked_ijk_dense")
REFERENCE_SIZE = 5
RTOL = 1e-9
ATOL = 1e-12


def load_matrix(path: Path, size: int) -> np.ndarray:
    raw = path.read_bytes()
    expected = size * size * 8
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype="<f8").reshape(size, size)


def run_jar(jar: Path, method: str, inputs: Path, size: int,
            density: int | None = None) -> float:
    """Run one configuration and return the checksum printed by the runner."""
    with tempfile.TemporaryDirectory(prefix="correctness_") as tmp:
        tmp_path = Path(tmp)
        command = ["java", "-jar", str(jar), "--method", method,
                   "--inputs", str(inputs), "--size", str(size),
                   "--memory-csv", str(tmp_path / "memory.csv")]
        if density is not None:
            command += ["--density", str(density),
                        "--sparse-time-csv", str(tmp_path / "time.csv")]
        else:
            command += ["--time-csv", str(tmp_path / "time.csv")]
        result = subprocess.run(command, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"jar failed ({result.returncode}): {' '.join(command)}\n"
                         f"{result.stdout}")
    return parse_checksum(result.stdout)


def parse_checksum(stdout: str) -> float:
    for line in reversed(stdout.strip().splitlines()):
        if "checksum=" in line:
            return float(line.split("checksum=")[1])
    raise SystemExit(f"no checksum in runner output:\n{stdout}")


def check(name: str, actual: float, expected: float) -> bool:
    ok = np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
    status = "ok" if ok else "MISMATCH"
    print(f"  [{status:>8}] {name}: jar={actual:.17g} numpy={expected:.17g}")
    return ok


def check_dense_methods(jar: Path) -> bool:
    matrix_a = load_matrix(REFERENCE_DIR / f"matrixA_{REFERENCE_SIZE}.bin", REFERENCE_SIZE)
    matrix_b = load_matrix(REFERENCE_DIR / f"matrixB_{REFERENCE_SIZE}.bin", REFERENCE_SIZE)
    expected_checksum = float(np.matmul(matrix_a, matrix_b).sum())
    print(f"dense reference n={REFERENCE_SIZE} (numpy checksum={expected_checksum:.17g})")
    return all(
        check(method, run_jar(jar, method, REFERENCE_DIR, REFERENCE_SIZE),
              expected_checksum)
        for method in DENSE_METHODS
    )


def check_sparse_method(jar: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="sparse_reference_") as tmp:
        inputs = Path(tmp)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "python" / "generate_sparse_inputs.py"),
             "--out-dir", str(inputs), "--sizes", "5", "--densities", "40"],
            check=True)
        matrix_a = load_matrix(inputs / "matrixA_n5_d40.bin", 5)
        matrix_b = load_matrix(inputs / "matrixB_n5_d40.bin", 5)
        expected_checksum = float(np.matmul(matrix_a, matrix_b).sum())
        print(f"sparse reference n=5 density=40% (numpy checksum={expected_checksum:.17g})")
        csr_checksum = run_jar(jar, "sparse_csr_gustavson", inputs, 5, density=40)
        dense_checksum = run_jar(jar, "triple_loop_ijk_dense", inputs, 5, density=40)
        return (check("sparse_csr_gustavson", csr_checksum, expected_checksum)
                and check("triple_loop_ijk_dense (sparse inputs)",
                          dense_checksum, expected_checksum))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar", type=Path,
                        default=PROJECT_ROOT / "target" / "matrixmul-optimised.jar")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.jar.is_file():
        print(f"jar not found: {args.jar}; run 'mvn -q package' first", file=sys.stderr)
        return 1
    ok = check_dense_methods(args.jar)
    ok = check_sparse_method(args.jar) and ok
    if not ok:
        print("correctness check FAILED", file=sys.stderr)
        return 1
    print("correctness check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
