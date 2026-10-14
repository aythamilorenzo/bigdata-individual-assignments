"""Campaign orchestrator for assignment 2 (dense optimisation + sparse CSR).

Builds the shaded jar once, then executes every declared configuration with
the shared CLI contract, logs every subprocess to run_log.txt, and validates
the raw CSVs before the analysis scripts consume them.

Part A runs each dense method (baseline i-j-k, i-k-j, cache-blocked) on the
shared dense inputs of assignment 1. Part B runs each sparse configuration
(the CSR kernel and the matched dense comparison on the same logical
matrices) on the declared-density inputs of generate_sparse_inputs.py. The
dense comparison above --dense-comparison-max-size runs at every density;
above that size it runs only at --dense-comparison-probe-density, because
dense time does not depend on the density of the entries (dense kernels do
not skip zeros), so one measured probe per size is representative and the
others would only repeat the same measurement.

Exit codes of the runner: 0 ok, 2 invalid arguments, 3 time budget exceeded
(a result, not a failure), 4 missing input files.

    run_experiments.py [--part a|b|all] [--sizes 10,25,...]
                       [--sparse-sizes 100,500,...] [--densities 1,5,...]
                       [--warmup 3] [--repetitions 7] [--append] [--skip-build]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DENSE_SIZES = [10, 25, 50, 100, 200, 500, 1000, 1500]
DEFAULT_SPARSE_SIZES = [100, 500, 1000, 2000]
DEFAULT_DENSITIES = [1, 5, 10, 20, 50]
DENSE_METHODS = ("triple_loop_ijk_dense", "triple_loop_ikj_dense", "blocked_ijk_dense")
SPARSE_METHOD = "sparse_csr_gustavson"
DEFAULT_DENSE_COMPARISON_METHOD = "triple_loop_ikj_dense"
DENSE_COMPARISON_MAX_SIZE = 1000
DENSE_COMPARISON_PROBE_DENSITY = 10

EXIT_OK = 0
EXIT_BUDGET_EXCEEDED = 3
EXIT_MISSING_INPUT = 4
JAR = PROJECT_ROOT / "target" / "matrixmul-optimised.jar"


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


def runner_command() -> list[str]:
    return ["java", "-jar", str(JAR)]


def part_a_command(size: int, method: str, time_csv: Path, memory_csv: Path,
                   warmup: int, repetitions: int) -> list[str]:
    return runner_command() + [
        "--method", method,
        "--inputs", str(PROJECT_ROOT / "data" / "inputs"),
        "--size", str(size),
        "--time-csv", str(time_csv),
        "--memory-csv", str(memory_csv),
        "--warmup", str(warmup),
        "--repetitions", str(repetitions),
    ]


def part_b_command(size: int, density: int, method: str, sparse_time_csv: Path,
                   memory_csv: Path, warmup: int, repetitions: int) -> list[str]:
    return runner_command() + [
        "--method", method,
        "--density", str(density),
        "--inputs", str(PROJECT_ROOT / "data" / "sparse"),
        "--size", str(size),
        "--sparse-time-csv", str(sparse_time_csv),
        "--memory-csv", str(memory_csv),
        "--warmup", str(warmup),
        "--repetitions", str(repetitions),
    ]


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


def report_part_a_anomalies(time_csv: Path, memory_csv: Path, sizes: list[int],
                            warmups: int, repetitions: int) -> None:
    time_rows = count_rows(time_csv)
    memory_rows = count_rows(memory_csv)
    print("\n=== Part A summary ===")
    anomalies = 0
    for method in DENSE_METHODS:
        for size in sizes:
            timed = [r for r in time_rows
                     if r["method"] == method and int(r["size"]) == size]
            measured = [r for r in timed if r["warmup"] == "false"]
            note = ""
            if len(measured) == 0:
                note = " (no measured rows: budget exceeded or failure)"
                anomalies += 1
            elif len(timed) != warmups + repetitions:
                note = f" (unexpected: {len(timed)} rows)"
                anomalies += 1
            print(f"{method:>26} n={size:<5} time_rows={len(timed):<3}{note}")
    print(f"total Part A time rows: {len(time_rows)}, "
          f"memory rows: {len(memory_rows)}")
    if anomalies:
        print(f"ANOMALIES: {anomalies} configuration(s) flagged above")


def report_part_b_anomalies(sparse_time_csv: Path, memory_csv: Path,
                            sizes: list[int], densities: list[int],
                            dense_comparison_method: str,
                            warmups: int, repetitions: int) -> None:
    time_rows = count_rows(sparse_time_csv)
    memory_rows = count_rows(memory_csv)
    print("\n=== Part B summary ===")
    anomalies = 0
    for size in sizes:
        for density in densities:
            csr_rows = [r for r in time_rows
                        if int(r["size"]) == size and int(r["density"]) == density
                        and r["representation"] == "csr"]
            dense_rows = [r for r in time_rows
                          if int(r["size"]) == size and int(r["density"]) == density
                          and r["representation"] == "dense"]
            note = ""
            if len(csr_rows) not in (0, warmups + repetitions):
                note = f" (unexpected csr rows: {len(csr_rows)})"
                anomalies += 1
            if len(csr_rows) == 0:
                note += " (no csr rows: budget exceeded or failure)"
                anomalies += 1
            if dense_rows and dense_rows[0]["method"] != dense_comparison_method:
                note += f" (dense method {dense_rows[0]['method']}?)"
                anomalies += 1
            print(f"n={size:<5} d={density:<3}% csr_rows={len(csr_rows):<3} "
                  f"dense_rows={len(dense_rows):<3}{note}")
    print(f"total Part B time rows: {len(time_rows)}, "
          f"memory rows: {len(memory_rows)}")
    if anomalies:
        print(f"ANOMALIES: {anomalies} configuration(s) flagged above")


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("a", "b", "all"), default="all",
                        help="which campaign part to run (default: both)")
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_DENSE_SIZES)),
                        help="comma-separated Part A dense sizes")
    parser.add_argument("--sparse-sizes",
                        default=",".join(map(str, DEFAULT_SPARSE_SIZES)),
                        help="comma-separated Part B sizes")
    parser.add_argument("--densities",
                        default=",".join(map(str, DEFAULT_DENSITIES)),
                        help="comma-separated Part B densities in percent")
    parser.add_argument("--dense-comparison-method",
                        default=DEFAULT_DENSE_COMPARISON_METHOD,
                        help="dense method used for the matched Part B comparison")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "data" / "results")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="wall-clock timeout per configuration, seconds")
    parser.add_argument("--append", action="store_true",
                        help="append to existing CSVs instead of starting fresh")
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    inputs_dir = PROJECT_ROOT / "data" / "inputs"
    if args.part in ("a", "all"):
        for size in parse_int_list(args.sizes):
            for side in ("A", "B"):
                path = inputs_dir / f"matrix{side}_{size}.bin"
                if not path.is_file():
                    raise SystemExit(
                        f"missing dense input {path}; "
                        "run python/generate_inputs.py first")
    if args.part in ("b", "all"):
        sparse_dir = PROJECT_ROOT / "data" / "sparse"
        for size in parse_int_list(args.sparse_sizes):
            for density in parse_int_list(args.densities):
                for side in ("A", "B"):
                    path = sparse_dir / f"matrix{side}_n{size}_d{density}.bin"
                    if not path.is_file():
                        raise SystemExit(
                            f"missing sparse input {path}; "
                            "run python/generate_sparse_inputs.py first")


def run_part_a(args: argparse.Namespace, handle) -> None:
    sizes = parse_int_list(args.sizes)
    time_csv = args.results / "dense_time_raw.csv"
    memory_csv = args.results / "memory_raw.csv"
    for size in sizes:
        for method in DENSE_METHODS:
            command = part_a_command(size, method, time_csv, memory_csv,
                                     args.warmup, args.repetitions)
            execute_configuration(command, handle, args.timeout)
    report_part_a_anomalies(time_csv, memory_csv, sizes,
                            args.warmup, args.repetitions)


def run_part_b(args: argparse.Namespace, handle) -> None:
    sizes = parse_int_list(args.sparse_sizes)
    densities = parse_int_list(args.densities)
    sparse_time_csv = args.results / "sparse_time_raw.csv"
    memory_csv = args.results / "memory_raw.csv"
    largest_size = max(sizes)
    for size in sizes:
        for density in densities:
            command = part_b_command(size, density, SPARSE_METHOD, sparse_time_csv,
                                     memory_csv, args.warmup, args.repetitions)
            execute_configuration(command, handle, args.timeout)
            if size < largest_size or density == DENSE_COMPARISON_PROBE_DENSITY:
                command = part_b_command(size, density, args.dense_comparison_method,
                                         sparse_time_csv, memory_csv,
                                         args.warmup, args.repetitions)
                execute_configuration(command, handle, args.timeout)
            else:
                log(handle, f"  dense comparison at n={size} d={density}% skipped: "
                            f"representative probe already measured at "
                            f"d={DENSE_COMPARISON_PROBE_DENSITY}%")
    report_part_b_anomalies(sparse_time_csv, memory_csv, sizes, densities,
                            args.dense_comparison_method,
                            args.warmup, args.repetitions)


def main() -> int:
    args = parse_args()
    if args.dense_comparison_method not in DENSE_METHODS:
        raise SystemExit(f"unknown dense comparison method: {args.dense_comparison_method}")
    require_inputs(args)

    args.results.mkdir(parents=True, exist_ok=True)
    memory_csv = args.results / "memory_raw.csv"
    if not args.append:
        (args.results / "dense_time_raw.csv").unlink(missing_ok=True)
        (args.results / "sparse_time_raw.csv").unlink(missing_ok=True)
        memory_csv.unlink(missing_ok=True)

    log_path = args.results / "run_log.txt"
    with log_path.open("a" if args.append else "w", encoding="utf-8") as handle:
        log(handle, f"campaign start: part={args.part} "
                    f"sizes={args.sizes} sparse_sizes={args.sparse_sizes} "
                    f"densities={args.densities} warmup={args.warmup} "
                    f"repetitions={args.repetitions} append={args.append}")
        if not JAR.is_file():
            build_jar(handle)
        if args.part in ("a", "all"):
            run_part_a(args, handle)
        if args.part in ("b", "all"):
            run_part_b(args, handle)
        log(handle, "campaign finished without failures")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
