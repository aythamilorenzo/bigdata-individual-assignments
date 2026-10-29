"""Campaign orchestrator for assignment 3 (thread-count scaling).

Builds the shaded jar once, then executes every declared configuration with
the shared CLI contract, logs every subprocess to run_log.txt, and validates
the raw CSVs before the analysis scripts consume them.

For each size the sequential baseline (the same cache-blocked kernel, one
thread) is measured once, then the parallel distribution of the same kernel
is measured at every declared thread count, including one thread, which
reveals the per-multiplication dispatch and join overhead of the framework.

Exit codes of the runner: 0 ok, 2 invalid arguments, 3 time budget exceeded
(a result, not a failure), 4 missing input files.

    run_experiments.py [--sizes 500,1000,...] [--threads 1,2,...]
                       [--warmup 3] [--repetitions 7] [--append] [--skip-build]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIZES = [500, 1000, 1500, 2000]
DEFAULT_THREADS = [1, 2, 4, 6, 8, 12, 24]
SEQUENTIAL_METHOD = "blocked_ijk_sequential"
PARALLEL_METHOD = "blocked_ijk_parallel"

EXIT_OK = 0
EXIT_BUDGET_EXCEEDED = 3
EXIT_MISSING_INPUT = 4
JAR = PROJECT_ROOT / "target" / "matrixmul-parallel.jar"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(handle, message: str) -> None:
    line = f"[{utc_timestamp()}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def run_checked(command: list[str], handle, cwd: Path | None = None) -> None:
    result = subprocess.run(
        command, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log(handle, f"$ {' '.join(command)}\n{result.stdout.strip()}")
    if result.returncode != 0:
        raise SystemExit(f"build step failed: {' '.join(command)}")


def build_jar(handle) -> None:
    run_checked(["mvn", "-q", "-DskipTests", "package"], handle, cwd=PROJECT_ROOT)


def configuration_command(size: int, method: str, threads: int, time_csv: Path,
                          memory_csv: Path, warmup: int, repetitions: int) -> list[str]:
    return ["java", "-jar", str(JAR),
            "--method", method,
            "--threads", str(threads),
            "--inputs", str(PROJECT_ROOT / "data" / "inputs"),
            "--size", str(size),
            "--time-csv", str(time_csv),
            "--memory-csv", str(memory_csv),
            "--warmup", str(warmup),
            "--repetitions", str(repetitions)]


def execute_configuration(command: list[str], handle, timeout_s: int) -> int:
    log(handle, f"$ {' '.join(command)}")
    result = subprocess.run(
        command, text=True, timeout=timeout_s,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in result.stdout.strip().splitlines():
        handle.write(f"    {line}\n")
    handle.flush()
    last_line = result.stdout.strip().splitlines()
    summary = last_line[-1] if last_line else "(no output)"
    if result.returncode == EXIT_OK:
        log(handle, f"  ok (exit 0): {summary}")
    elif result.returncode == EXIT_BUDGET_EXCEEDED:
        log(handle, "  time budget exceeded (exit 3): recorded as a result")
    elif result.returncode == EXIT_MISSING_INPUT:
        raise SystemExit(f"missing input files for: {' '.join(command)}")
    else:
        raise SystemExit(f"unexpected exit {result.returncode} for: {' '.join(command)}")
    return result.returncode


def count_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def report_anomalies(time_csv: Path, memory_csv: Path, sizes: list[int],
                     threads: list[int], warmups: int, repetitions: int) -> None:
    time_rows = count_rows(time_csv)
    memory_rows = count_rows(memory_csv)
    print("\n=== Campaign summary ===")
    anomalies = 0
    for size in sizes:
        configurations = [(SEQUENTIAL_METHOD, 1)] + [(PARALLEL_METHOD, p) for p in threads]
        for method, p in configurations:
            timed = [r for r in time_rows
                     if r["method"] == method and int(r["size"]) == size
                     and int(r["threads"]) == p]
            measured = [r for r in timed if r["warmup"] == "false"]
            note = ""
            if len(measured) == 0:
                note = " (no measured rows: budget exceeded or failure)"
                anomalies += 1
            elif len(timed) != warmups + repetitions:
                note = f" (unexpected: {len(timed)} rows)"
                anomalies += 1
            print(f"{method:>24} n={size:<5} p={p:<3} time_rows={len(timed):<3}{note}")
    print(f"total time rows: {len(time_rows)}, memory rows: {len(memory_rows)}")
    if anomalies:
        print(f"ANOMALIES: {anomalies} configuration(s) flagged above")


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)),
                        help="comma-separated matrix sizes")
    parser.add_argument("--threads", default=",".join(map(str, DEFAULT_THREADS)),
                        help="comma-separated thread counts for the parallel method")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "data" / "results")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="wall-clock timeout per configuration, seconds")
    parser.add_argument("--append", action="store_true",
                        help="append to existing CSVs instead of starting fresh")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse the jar already in target/ without rebuilding")
    return parser.parse_args()


def require_inputs(sizes: list[int]) -> None:
    inputs_dir = PROJECT_ROOT / "data" / "inputs"
    for size in sizes:
        for side in ("A", "B"):
            path = inputs_dir / f"matrix{side}_{size}.bin"
            if not path.is_file():
                raise SystemExit(
                    f"missing dense input {path}; "
                    "run python/generate_inputs.py first")


def run_campaign(args: argparse.Namespace, handle) -> None:
    sizes = parse_int_list(args.sizes)
    threads = parse_int_list(args.threads)
    time_csv = args.results / "parallel_time_raw.csv"
    memory_csv = args.results / "memory_raw.csv"
    for size in sizes:
        command = configuration_command(size, SEQUENTIAL_METHOD, 1, time_csv,
                                        memory_csv, args.warmup, args.repetitions)
        execute_configuration(command, handle, args.timeout)
        for p in threads:
            command = configuration_command(size, PARALLEL_METHOD, p, time_csv,
                                            memory_csv, args.warmup, args.repetitions)
            execute_configuration(command, handle, args.timeout)
    report_anomalies(time_csv, memory_csv, sizes, threads,
                     args.warmup, args.repetitions)


def main() -> int:
    args = parse_args()
    sizes = parse_int_list(args.sizes)
    require_inputs(sizes)

    args.results.mkdir(parents=True, exist_ok=True)
    if not args.append:
        (args.results / "parallel_time_raw.csv").unlink(missing_ok=True)
        (args.results / "memory_raw.csv").unlink(missing_ok=True)

    log_path = args.results / "run_log.txt"
    with log_path.open("a" if args.append else "w", encoding="utf-8") as handle:
        log(handle, f"campaign start: sizes={args.sizes} threads={args.threads} "
                    f"warmup={args.warmup} repetitions={args.repetitions} "
                    f"append={args.append}")
        if not args.skip_build:
            build_jar(handle)
        run_campaign(args, handle)
        log(handle, "campaign finished without failures")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
