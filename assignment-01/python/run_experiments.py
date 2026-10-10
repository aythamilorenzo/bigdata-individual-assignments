"""Campaign orchestrator for the four-language matrix multiplication benchmark.

Builds every benchmark runner once, then executes each configuration
(size x language) with the shared CLI contract, logs every subprocess to
run_log.txt, and validates the raw CSVs before the analysis scripts consume
them.

Exit codes of the runners: 0 ok, 2 invalid arguments, 3 time budget exceeded
(a result, not a failure), 4 missing input files.

    run_experiments.py [--sizes 10,25,...] [--languages python,java,c,go]
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
DEFAULT_SIZES = [10, 25, 50, 100, 200, 500, 1000]
ALL_LANGUAGES = ("python", "java", "c", "go")
TIME_HEADER = [
    "size", "language", "method", "numeric_type", "repetition", "warmup",
    "batch_size", "time_ns_per_multiply", "checksum",
]
MEMORY_HEADER = [
    "size", "language", "metric", "tool", "value_bytes", "theoretical_bytes",
]
EXIT_OK = 0
EXIT_BUDGET_EXCEEDED = 3
EXIT_MISSING_INPUT = 4


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


def build_runners(handle, build_dir: Path) -> dict[str, list[str]]:
    run_checked(["mvn", "-q", "-DskipTests", "package"], handle, cwd=PROJECT_ROOT)
    run_checked(["bash", "c/build.sh"], handle, cwd=PROJECT_ROOT)
    go_binary = build_dir / "go-benchmark"
    run_checked(["go", "build", "-o", str(go_binary), "./cmd/benchmark"],
                handle, cwd=PROJECT_ROOT / "go")
    return {
        "python": [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(PROJECT_ROOT / "python" / "run_benchmark.py"),
        ],
        "java": [
            "java", "-jar",
            str(PROJECT_ROOT / "target" / "matrixmul-benchmark.jar"),
        ],
        "c": [str(PROJECT_ROOT / "c" / "matrixmul_benchmark")],
        "go": [str(go_binary)],
    }


def configuration_command(runners: dict[str, list[str]], language: str,
                          size: int, inputs: Path, time_csv: Path,
                          memory_csv: Path, warmup: int,
                          repetitions: int) -> list[str]:
    return runners[language] + [
        "--inputs", str(inputs),
        "--size", str(size),
        "--time-csv", str(time_csv),
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
        raise SystemExit(f"missing input files for n={' '.join(command)}")
    else:
        raise SystemExit(f"unexpected exit {result.returncode} for: {' '.join(command)}")
    return result.returncode


def count_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def report_anomalies(time_csv: Path, memory_csv: Path, sizes: list[int],
                     languages: list[str], warmups: int, repetitions: int) -> None:
    time_rows = count_rows(time_csv)
    memory_rows = count_rows(memory_csv)
    print("\n=== campaign summary ===")
    anomalies = 0
    for language in languages:
        for size in sizes:
            timed = [r for r in time_rows
                     if r["language"] == language and int(r["size"]) == size]
            mem = [r for r in memory_rows
                   if r["language"] == language and int(r["size"]) == size]
            measured = [r for r in timed if r["warmup"] == "false"]
            note = ""
            if len(measured) == 0:
                note = " (no measured rows: budget exceeded or failure)"
                anomalies += 1
            elif len(measured) != repetitions or len(timed) != warmups + repetitions:
                note = f" (unexpected: {len(timed)} rows)"
                anomalies += 1
            if not mem:
                note += " (missing memory rows)"
                anomalies += 1
            print(f"{language:>6} n={size:<5} time_rows={len(timed):<3} "
                  f"memory_rows={len(mem)}{note}")
    print(f"total time rows: {len(time_rows)}, memory rows: {len(memory_rows)}")
    if anomalies:
        print(f"ANOMALIES: {anomalies} configuration(s) flagged above")
    else:
        print("no anomalies: every configuration has the expected rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)),
                        help="comma-separated matrix sizes")
    parser.add_argument("--languages", default=",".join(ALL_LANGUAGES),
                        help="comma-separated languages to run")
    parser.add_argument("--inputs", type=Path, default=PROJECT_ROOT / "data" / "inputs")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "data" / "results")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=900,
                        help="wall-clock timeout per configuration, seconds")
    parser.add_argument("--append", action="store_true",
                        help="append to existing CSVs instead of starting fresh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sizes = [int(text) for text in args.sizes.split(",")]
    languages = [text.strip() for text in args.languages.split(",")]
    unknown = set(languages) - set(ALL_LANGUAGES)
    if unknown:
        raise SystemExit(f"unknown languages: {sorted(unknown)}")
    if not args.inputs.is_dir():
        raise SystemExit(f"inputs directory not found: {args.inputs}")

    args.results.mkdir(parents=True, exist_ok=True)
    time_csv = args.results / "time_raw.csv"
    memory_csv = args.results / "memory_raw.csv"
    if not args.append:
        time_csv.unlink(missing_ok=True)
        memory_csv.unlink(missing_ok=True)

    build_dir = Path(tempfile.mkdtemp(prefix="matrixmul_build_"))
    log_path = args.results / "run_log.txt"
    with log_path.open("a" if args.append else "w", encoding="utf-8") as handle:
        log(handle, f"campaign start: sizes={sizes} languages={languages} "
                    f"warmup={args.warmup} repetitions={args.repetitions} "
                    f"append={args.append}")
        try:
            runners = build_runners(handle, build_dir)
            for size in sizes:
                for language in languages:
                    command = configuration_command(
                        runners, language, size, args.inputs,
                        time_csv, memory_csv, args.warmup, args.repetitions)
                    execute_configuration(command, handle, args.timeout)
            log(handle, "campaign finished without failures")
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    report_anomalies(time_csv, memory_csv, sizes, languages,
                     args.warmup, args.repetitions)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
