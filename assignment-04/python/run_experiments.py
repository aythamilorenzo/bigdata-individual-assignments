#!/usr/bin/env python3
"""Campaign orchestrator for Assignment 4 — distributed matrix multiplication.

Runs one JVM per (method, threads, size) configuration so that the peak-RSS
and peak-heap measurements of each configuration are not polluted by the
others. The three method families share the same validated kernel:

- blocked_ijk_sequential  (threads = 1)
- blocked_ijk_parallel    (thread pool, assignment 3)
- blocked_ijk_distributed (Spark local[P], assignment 4)

Raw rows are appended to data/results/distributed_time_raw.csv and
data/results/memory_raw.csv.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JAVA_CMD = "java"
JAR_PATH = PROJECT_ROOT / "target" / "matrixmul-distributed.jar"
INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
TIME_CSV = RESULTS_DIR / "distributed_time_raw.csv"
MEMORY_CSV = RESULTS_DIR / "memory_raw.csv"
LOG_PATH = RESULTS_DIR / "run_log.txt"

# Spark on Java 17 needs these module opens on every JVM.
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

SIZES = (500, 1000, 1500, 2000, 3000, 4000)
THREAD_COUNTS = (1, 2, 4, 6, 8, 12, 24)
WARMUP = 3
REPETITIONS = 7


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_configuration(method: str, threads: int, size: int) -> bool:
    command = [JAVA_CMD, *JVM_ARGS, "-Xmx3g", "-jar", str(JAR_PATH),
               "--method", method,
               "--threads", str(threads),
               "--inputs", str(INPUTS_DIR),
               "--size", str(size),
               "--time-csv", str(TIME_CSV),
               "--memory-csv", str(MEMORY_CSV),
               "--warmup", str(WARMUP),
               "--repetitions", str(REPETITIONS)]
    log(f"$ {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    for line in result.stdout.splitlines():
        log(f"  {line}")
    if result.returncode != 0:
        for line in result.stderr.splitlines()[-5:]:
            log(f"  ! {line}")
        log(f"  exit {result.returncode} (3 = budget exceeded, a result not a failure)")
    return result.returncode in (0, 3)


def main() -> int:
    if not JAR_PATH.is_file():
        print(f"missing jar: {JAR_PATH}; run 'mvn -q -DskipTests package' first")
        return 1

    log("=== Assignment 4 campaign: sequential vs thread-pool vs Spark ===")
    log(f"sizes={SIZES} threads={THREAD_COUNTS} warmup={WARMUP} repetitions={REPETITIONS}")

    failures = 0
    total = 0
    for size in SIZES:
        total += 1
        ok = run_configuration("blocked_ijk_sequential", 1, size)
        failures += 0 if ok else 1
        for threads in THREAD_COUNTS:
            for method in ("blocked_ijk_parallel", "blocked_ijk_distributed"):
                total += 1
                ok = run_configuration(method, threads, size)
                failures += 0 if ok else 1

    log(f"=== campaign finished: {total} configurations, {failures} failures ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
