"""Benchmark runner for the Python triple-loop implementation.

Measures the multiplication kernel only: inputs are loaded and the output
buffer allocated before the timed region; each timed sample is either one
multiplication or a batch of K multiplications (when a single multiplication
takes less than 2 ms), reported as time per multiplication. Warm-up samples
are recorded with warmup=true and excluded from the analysis.

CLI contract shared by the Python, Java, C and Go runners:

    run_benchmark --inputs DIR --size N --time-csv FILE --memory-csv FILE
                  --warmup W --repetitions R
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from pathlib import Path

from memory_metrics import peak_rss_bytes
from triple_loop_mul import multiply_into, zero_matrix

LANGUAGE = "python"
METHOD = "triple_loop_ijk_dense"
NUMERIC_TYPE = "float64"
BATCH_THRESHOLD_NS = 2_000_000
BUDGET_NS = 60_000_000_000
MAX_BATCH = 100_000

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


def load_matrix(path: Path, size: int) -> list[list[float]]:
    raw = path.read_bytes()
    expected = size * size * 8
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(raw)}")
    flat = list(struct.unpack(f"<{size * size}d", raw))
    return [flat[i * size:(i + 1) * size] for i in range(size)]


def checksum_row_major(out: list[list[float]]) -> float:
    total = 0.0
    for row in out:
        for value in row:
            total += value
    return total


def timed_multiply(a: list[list[float]], b: list[list[float]],
                   out: list[list[float]]) -> int:
    start = time.perf_counter_ns()
    multiply_into(a, b, out)
    return time.perf_counter_ns() - start


def append_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(header)
        writer.writerows(rows)


def time_row(size: int, repetition: int, warmup: bool, batch_size: int,
             time_ns_per_multiply: float, checksum: float) -> list[object]:
    return [
        size, LANGUAGE, METHOD, NUMERIC_TYPE, repetition, str(warmup).lower(),
        batch_size, f"{time_ns_per_multiply:.3f}", f"{checksum:.17g}",
    ]


def measure(a: list[list[float]], b: list[list[float]], out: list[list[float]],
            size: int, warmups: int, repetitions: int
            ) -> tuple[list[list[object]], int]:
    probe_ns = timed_multiply(a, b, out)
    if probe_ns > BUDGET_NS:
        raise BudgetExceeded(probe_ns)
    batch_size = 1 if probe_ns >= BATCH_THRESHOLD_NS else min(
        MAX_BATCH, max(1, -(-BATCH_THRESHOLD_NS // probe_ns))
    )

    rows: list[list[object]] = []
    for repetition in range(1, warmups + 1):
        total_ns = sum(timed_multiply(a, b, out) for _ in range(batch_size))
        rows.append(time_row(size, repetition, True, batch_size,
                             total_ns / batch_size, 0.0))
    for repetition in range(1, repetitions + 1):
        total_ns = sum(timed_multiply(a, b, out) for _ in range(batch_size))
        rows.append(time_row(size, repetition, False, batch_size,
                             total_ns / batch_size, 0.0))
    return rows, batch_size


def memory_rows(size: int, peak_rss: int) -> list[list[object]]:
    theoretical = 24 * size * size
    return [
        [size, LANGUAGE, "peak_rss", "/proc/self/status VmHWM",
         peak_rss, theoretical],
        [size, LANGUAGE, "theoretical_array_storage", "analytical",
         theoretical, theoretical],
    ]


class BudgetExceeded(Exception):
    def __init__(self, probe_ns: int) -> None:
        super().__init__(f"single multiplication took {probe_ns / 1e9:.1f} s")
        self.probe_ns = probe_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--time-csv", type=Path, required=True)
    parser.add_argument("--memory-csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path_a = args.inputs / f"matrixA_{args.size}.bin"
    path_b = args.inputs / f"matrixB_{args.size}.bin"
    if not path_a.exists() or not path_b.exists():
        print(f"missing input files for n={args.size}", file=sys.stderr)
        return EXIT_MISSING_INPUT

    matrix_a = load_matrix(path_a, args.size)
    matrix_b = load_matrix(path_b, args.size)
    out = zero_matrix(args.size)
    try:
        rows, batch_size = measure(matrix_a, matrix_b, out, args.size,
                                   args.warmup, args.repetitions)
    except BudgetExceeded as exceeded:
        print(f"budget exceeded at n={args.size}: {exceeded}", file=sys.stderr)
        return EXIT_BUDGET_EXCEEDED

    checksum = checksum_row_major(out)
    for row in rows:
        row[-1] = f"{checksum:.17g}"
    append_rows(args.time_csv, TIME_HEADER, rows)
    append_rows(args.memory_csv, MEMORY_HEADER,
                memory_rows(args.size, peak_rss_bytes()))
    print(f"python n={args.size} batch={batch_size} checksum={checksum:.17g}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
