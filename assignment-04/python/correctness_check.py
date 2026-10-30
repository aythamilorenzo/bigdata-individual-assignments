#!/usr/bin/env python3
"""Correctness cross-check for Assignment 4.

Runs the compiled jar with every method family (sequential, thread pool,
Spark distributed) on the 5x5 reference pair and the shared campaign inputs,
and compares the printed checksums against a NumPy float64 reference
(rtol=1e-9, atol=1e-12). Also asserts that the checksum is invariant with
the thread count — evidence of safe partitioned access to the shared output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JAR_PATH = PROJECT_ROOT / "target" / "matrixmul-distributed.jar"
INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

JVM_ARGS = [
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
]

RTOL = 1e-9
ATOL = 1e-12


def load_matrix(path: Path, size: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape(size, size)


def run_jar_checksum(method: str, threads: int, inputs: Path, size: int) -> float:
    command = ["java", *JVM_ARGS, "-Xmx3g", "-jar", str(JAR_PATH),
               "--method", method, "--threads", str(threads),
               "--inputs", str(inputs), "--size", str(size),
               "--time-csv", "/tmp/correctness_time.csv",
               "--memory-csv", "/tmp/correctness_mem.csv",
               "--warmup", "0", "--repetitions", "1"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        raise RuntimeError(f"jar failed ({result.returncode}): {result.stderr[-300:]}")
    for line in result.stdout.splitlines():
        if "checksum=" in line:
            return float(line.split("checksum=")[1].split()[0])
    raise RuntimeError("checksum not found in output")


def check_case(label: str, inputs: Path, size: int,
               methods_threads: list[tuple[str, int]]) -> bool:
    print(f"\n{label}")
    a = load_matrix(inputs / f"matrixA_{size}.bin", size)
    b = load_matrix(inputs / f"matrixB_{size}.bin", size)
    reference = float(np.sum(a @ b))

    ok = True
    checksums = {}
    for method, threads in methods_threads:
        jar = run_jar_checksum(method, threads, inputs, size)
        checksums[(method, threads)] = jar
        close = np.isclose(jar, reference, rtol=RTOL, atol=ATOL)
        status = "ok" if close else "FAIL"
        print(f"  [{status:>4}] {method} p={threads}: jar={jar!r} numpy={reference!r}")
        ok = ok and close

    distinct = {round(v, 9) for v in checksums.values()}
    if len(distinct) != 1:
        print(f"  [FAIL] checksum varies across methods/threads: {distinct}")
        ok = False
    else:
        print("  [   ok] checksum invariant across all methods and thread counts")
    return ok


def main() -> int:
    if not JAR_PATH.is_file():
        print(f"missing jar: {JAR_PATH}")
        return 1

    cases = []
    cases.append(check_case(
        "reference n=5 (assignment 1 matrices)", REFERENCE_DIR, 5,
        [("blocked_ijk_sequential", 1),
         ("blocked_ijk_parallel", 2),
         ("blocked_ijk_parallel", 8),
         ("blocked_ijk_distributed", 2),
         ("blocked_ijk_distributed", 8)]))
    cases.append(check_case(
        "campaign input n=500", INPUTS_DIR, 500,
        [("blocked_ijk_sequential", 1),
         ("blocked_ijk_parallel", 1),
         ("blocked_ijk_parallel", 12),
         ("blocked_ijk_distributed", 1),
         ("blocked_ijk_distributed", 12)]))

    passed = all(cases)
    print(f"\n{'all checks passed' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
