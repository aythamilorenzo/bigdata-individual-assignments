"""Cross-check the compiled jar against a NumPy reference on known matrices.

The JUnit suite already validates the parallel kernel against the sequential
baseline in-JVM across sizes and thread counts. This script adds an
independent, out-of-JVM check: it runs the shaded jar on the 5x5 reference
matrices of assignments 1 and 2 (and on a larger shared input) with the
sequential method and with the parallel method at several thread counts, and
compares the checksums printed by the runner with NumPy float64 results.

Tolerances: rtol=1e-9, atol=1e-12 — the same tolerances as assignments 1
and 2 — because thread partitioning changes neither the summation order nor
the arithmetic, but the JVM and NumPy may still contract or schedule FP
operations differently; a mismatch beyond tolerance would indicate a
partitioning bug, not rounding.

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
INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
REFERENCE_SIZE = 5
PARALLEL_METHOD = "blocked_ijk_parallel"
CHECK_THREAD_COUNTS = (1, 2, 3, 4, 8)
RTOL = 1e-9
ATOL = 1e-12


def load_matrix(path: Path, size: int) -> np.ndarray:
    raw = path.read_bytes()
    expected = size * size * 8
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype="<f8").reshape(size, size)


def run_jar(jar: Path, method: str, inputs: Path, size: int, threads: int) -> float:
    """Run one configuration and return the checksum printed by the runner."""
    with tempfile.TemporaryDirectory(prefix="correctness_") as tmp:
        tmp_path = Path(tmp)
        command = ["java", "-jar", str(jar), "--method", method,
                   "--threads", str(threads),
                   "--inputs", str(inputs), "--size", str(size),
                   "--time-csv", str(tmp_path / "time.csv"),
                   "--memory-csv", str(tmp_path / "memory.csv")]
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


def check_configuration(jar: Path, label: str, inputs: Path, size: int,
                        method: str, threads: int) -> bool:
    matrix_a = load_matrix(inputs / f"matrixA_{size}.bin", size)
    matrix_b = load_matrix(inputs / f"matrixB_{size}.bin", size)
    expected_checksum = float(np.matmul(matrix_a, matrix_b).sum())
    actual = run_jar(jar, method, inputs, size, threads)
    return check(f"{label} {method} p={threads}", actual, expected_checksum)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar", type=Path,
                        default=PROJECT_ROOT / "target" / "matrixmul-parallel.jar")
    args = parser.parse_args()
    if not args.jar.is_file():
        print(f"jar not found: {args.jar}; run 'mvn -q package' first", file=sys.stderr)
        return 1

    print(f"reference n={REFERENCE_SIZE} (assignment 1 matrices)")
    ok = all(
        check_configuration(args.jar, "reference", REFERENCE_DIR, REFERENCE_SIZE,
                            method, threads)
        for method in ("blocked_ijk_sequential", PARALLEL_METHOD)
        for threads in ((1,) if method == "blocked_ijk_sequential"
                        else CHECK_THREAD_COUNTS)
    )

    shared_size = 500
    shared_inputs = INPUTS_DIR
    if (shared_inputs / f"matrixA_{shared_size}.bin").is_file():
        print(f"shared campaign input n={shared_size}")
        ok = check_configuration(args.jar, "campaign", shared_inputs, shared_size,
                                 "blocked_ijk_sequential", 1) and ok
        ok = all(
            check_configuration(args.jar, "campaign", shared_inputs, shared_size,
                                PARALLEL_METHOD, threads)
            for threads in CHECK_THREAD_COUNTS
        ) and ok
    else:
        print(f"shared input n={shared_size} not generated; skipping the campaign check")

    if not ok:
        print("correctness check FAILED", file=sys.stderr)
        return 1
    print("correctness check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
