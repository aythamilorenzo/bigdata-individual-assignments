"""Plot the assignment 3 campaign: thread-count scaling of one dense kernel.

Every figure is regenerated from the raw CSVs alone and written as PDF into
report/figures/ for inclusion in the LaTeX report:

- median_time_vs_threads.pdf   kernel time per configuration vs p (log-log)
- speedup_vs_threads.pdf       S(p) = Tseq / Tp per size, ideal line S(p) = p
- efficiency_vs_threads.pdf    E(p) = S(p) / p per size
- framework_overhead.pdf       parallel p=1 vs the sequential baseline
- peak_rss_vs_threads.pdf      peak resident memory per configuration

Speedup and efficiency use the median as the summary statistic (declared in
the report): Tseq(n) is the median of the sequential baseline at size n, and
Tp(n, p) the median of the parallel method at size n and p threads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

SEQUENTIAL_METHOD = "blocked_ijk_sequential"
PARALLEL_METHOD = "blocked_ijk_parallel"
HARDWARE_THREADS = 12
PHYSICAL_CORES = 6


def measured(df: pd.DataFrame) -> pd.DataFrame:
    """Keep measured (non warm-up) samples only.

    pandas may parse the warmup column either as bool or as string, so the
    filter accepts both spellings.
    """
    warmup = df["warmup"]
    keep = warmup.map(lambda value: value in (False, "false"))
    return df[keep].copy()


def median_iqr(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Median and interquartile range of the kernel time per group."""
    stats = df.groupby(group)["time_ns_per_multiply"].agg(
        median="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
    ).reset_index()
    stats["iqr"] = stats["q3"] - stats["q1"]
    return stats


def plot_time_vs_threads(parallel: pd.DataFrame) -> None:
    stats = median_iqr(parallel, ["size", "threads"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for size in sorted(stats["size"].unique()):
        rows = stats[stats["size"] == size].sort_values("threads")
        ax.plot(rows["threads"], rows["median"] / 1e9, "o-", label=f"n = {size}")
        ax.fill_between(rows["threads"], rows["q1"] / 1e9, rows["q3"] / 1e9, alpha=0.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("threads p")
    ax.set_ylabel("kernel time per multiplication (s, median with IQR)")
    ax.set_title("Parallel blocked kernel: runtime vs thread count")
    ax.legend()
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "median_time_vs_threads.pdf")
    plt.close(fig)


def plot_speedup_vs_threads(sequential: pd.DataFrame, parallel: pd.DataFrame) -> None:
    base = median_iqr(sequential, ["size"])[["size", "median"]]
    base = base.rename(columns={"median": "seq_ns"})
    stats = median_iqr(parallel, ["size", "threads"])
    merged = stats.merge(base, on="size")
    merged["speedup"] = merged["seq_ns"] / merged["median"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for size in sorted(merged["size"].unique()):
        rows = merged[merged["size"] == size].sort_values("threads")
        ax.plot(rows["threads"], rows["speedup"], "o-", label=f"n = {size}")
    ideal = sorted(merged["threads"].unique())
    ax.plot(ideal, ideal, "k--", linewidth=1.0, label="ideal S(p) = p")
    ax.axvline(HARDWARE_THREADS, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("threads p")
    ax.set_ylabel("speedup S(p) = Tseq / Tp (median)")
    ax.set_title("Speedup vs thread count")
    ax.legend()
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "speedup_vs_threads.pdf")
    plt.close(fig)


def plot_efficiency_vs_threads(sequential: pd.DataFrame, parallel: pd.DataFrame) -> None:
    base = median_iqr(sequential, ["size"])[["size", "median"]]
    base = base.rename(columns={"median": "seq_ns"})
    stats = median_iqr(parallel, ["size", "threads"])
    merged = stats.merge(base, on="size")
    merged["efficiency"] = (merged["seq_ns"] / merged["median"]) / merged["threads"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for size in sorted(merged["size"].unique()):
        rows = merged[merged["size"] == size].sort_values("threads")
        ax.plot(rows["threads"], rows["efficiency"], "o-", label=f"n = {size}")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(HARDWARE_THREADS, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("threads p")
    ax.set_ylabel("efficiency E(p) = S(p) / p")
    ax.set_title("Parallel efficiency vs thread count")
    ax.legend()
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "efficiency_vs_threads.pdf")
    plt.close(fig)


def plot_framework_overhead(sequential: pd.DataFrame, parallel: pd.DataFrame) -> None:
    """Parallel p=1 versus the sequential baseline: the framework's own cost."""
    base = median_iqr(sequential, ["size"])[["size", "median"]]
    base = base.rename(columns={"median": "seq_ns"})
    one = median_iqr(parallel[parallel["threads"] == 1], ["size"])
    merged = one.merge(base, on="size")
    merged["overhead_pct"] = 100.0 * (merged["median"] / merged["seq_ns"] - 1.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(merged["size"].astype(str), merged["overhead_pct"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("p=1 parallel vs sequential (%)")
    ax.set_title("Framework overhead: parallel with one thread")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "framework_overhead.pdf")
    plt.close(fig)


def plot_peak_rss(memory: pd.DataFrame) -> None:
    peak_rss = memory[memory["metric"] == "peak_rss"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method, label in ((SEQUENTIAL_METHOD, "sequential"),
                          (PARALLEL_METHOD, "parallel")):
        rows = peak_rss[peak_rss["method"] == method]
        for size in sorted(rows["size"].unique()):
            part = rows[rows["size"] == size].sort_values("threads")
            ax.plot(part["threads"], part["value_bytes"] / 2**20, "o-",
                    label=f"{label} n = {size}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("threads p")
    ax.set_ylabel("peak RSS (MiB)")
    ax.set_title("Peak resident memory per configuration")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "peak_rss_vs_threads.pdf")
    plt.close(fig)


def main() -> int:
    time_path = RESULTS_DIR / "parallel_time_raw.csv"
    memory_path = RESULTS_DIR / "memory_raw.csv"
    for path in (time_path, memory_path):
        if not path.is_file():
            print(f"missing results file: {path}", file=sys.stderr)
            return 1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(time_path)
    sequential = measured(rows[rows["method"] == SEQUENTIAL_METHOD])
    parallel = measured(rows[rows["method"] == PARALLEL_METHOD])
    memory = pd.read_csv(memory_path)

    plot_time_vs_threads(parallel)
    plot_speedup_vs_threads(sequential, parallel)
    plot_efficiency_vs_threads(sequential, parallel)
    plot_framework_overhead(sequential, parallel)
    plot_peak_rss(memory)
    print(f"figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
