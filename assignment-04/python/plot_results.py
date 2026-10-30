#!/usr/bin/env python3
"""Figures for Assignment 4 from the raw campaign CSVs.

Six figures, all reproducible from data/results/*.csv alone:

1. median time vs threads (three method families, per size)
2. speedup vs threads (thread pool vs Spark, ideal line)
3. efficiency vs threads
4. framework overhead (Spark minus thread pool, per thread count)
5. Spark phase breakdown (broadcast / job / assemble, stacked)
6. peak memory (RSS and heap vs threads, per size)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.9,
})

COLORS = {
    "blocked_ijk_sequential": "#2c3e50",
    "blocked_ijk_parallel": "#2980b9",
    "blocked_ijk_distributed": "#c0392b",
}
LABELS = {
    "blocked_ijk_sequential": "sequential (thread pool = 1)",
    "blocked_ijk_parallel": "thread pool (assignment 3)",
    "blocked_ijk_distributed": "Spark local[p] (assignment 4)",
}


def load_times() -> pd.DataFrame:
    csv = RESULTS_DIR / "distributed_time_raw.csv"
    df = pd.read_csv(csv)
    measured = df[~df["warmup"]].copy()
    for column in ("end_to_end_ns", "broadcast_ns", "job_wall_ns",
                   "assemble_ns", "task_compute_ns"):
        measured[column.replace("_ns", "_ms")] = measured[column] / 1e6
    return measured


def load_memory() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "memory_raw.csv")


def medians(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["size", "threads", "method"], as_index=False)
    return grouped.agg(
        median_ms=("end_to_end_ms", "median"),
        q1=("end_to_end_ms", lambda x: x.quantile(0.25)),
        q3=("end_to_end_ms", lambda x: x.quantile(0.75)),
    )


def fig_time(med: pd.DataFrame) -> None:
    sizes = sorted(med["size"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, size in zip(axes.flat, sizes):
        for method in ("blocked_ijk_sequential", "blocked_ijk_parallel",
                       "blocked_ijk_distributed"):
            part = med[(med["size"] == size) & (med["method"] == method)]
            if part.empty:
                continue
            ax.plot(part["threads"], part["median_ms"], marker="o",
                    color=COLORS[method], label=LABELS[method])
            ax.fill_between(part["threads"], part["q1"], part["q3"],
                            color=COLORS[method], alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"n = {size}")
        ax.set_xlabel("threads p")
        ax.set_ylabel("median end-to-end time (ms)")
    for ax in axes.flat[len(sizes):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES_DIR / "median_time_vs_threads.png", dpi=150)
    plt.close(fig)


def fig_speedup(med: pd.DataFrame) -> None:
    sizes = sorted(med["size"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, size in zip(axes.flat, sizes):
        for method in ("blocked_ijk_parallel", "blocked_ijk_distributed"):
            part = med[(med["size"] == size) & (med["method"] == method)].sort_values("threads")
            seq = med[(med["size"] == size) & (med["method"] == "blocked_ijk_sequential")]
            if part.empty or seq.empty:
                continue
            seq_time = seq["median_ms"].iloc[0]
            speedup = seq_time / part["median_ms"]
            ax.plot(part["threads"], speedup, marker="o",
                    color=COLORS[method], label=LABELS[method])
        thread_axis = np.array(sorted(med["threads"].unique()))
        ax.plot(thread_axis, thread_axis, "k--", alpha=0.4, label="ideal S(p) = p")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"n = {size}")
        ax.set_xlabel("threads p")
        ax.set_ylabel("speedup S(p)")
    for ax in axes.flat[len(sizes):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES_DIR / "speedup_vs_threads.png", dpi=150)
    plt.close(fig)


def fig_efficiency(med: pd.DataFrame) -> None:
    sizes = sorted(med["size"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, size in zip(axes.flat, sizes):
        for method in ("blocked_ijk_parallel", "blocked_ijk_distributed"):
            part = med[(med["size"] == size) & (med["method"] == method)].sort_values("threads")
            seq = med[(med["size"] == size) & (med["method"] == "blocked_ijk_sequential")]
            if part.empty or seq.empty:
                continue
            seq_time = seq["median_ms"].iloc[0]
            efficiency = (seq_time / part["median_ms"]) / part["threads"]
            ax.plot(part["threads"], efficiency, marker="o",
                    color=COLORS[method], label=LABELS[method])
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="ideal E = 1")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"n = {size}")
        ax.set_xlabel("threads p")
        ax.set_ylabel("efficiency E(p)")
    for ax in axes.flat[len(sizes):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES_DIR / "efficiency_vs_threads.png", dpi=150)
    plt.close(fig)


def fig_overhead(med: pd.DataFrame) -> None:
    """Spark time minus thread-pool time, as a percentage of the thread-pool time."""
    merged = med[med["method"] == "blocked_ijk_distributed"].merge(
        med[med["method"] == "blocked_ijk_parallel"],
        on=["size", "threads"], suffixes=("_spark", "_pool"))
    merged["overhead_pct"] = 100 * (merged["median_ms_spark"] / merged["median_ms_pool"] - 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for size in sorted(merged["size"].unique()):
        part = merged[merged["size"] == size].sort_values("threads")
        ax.plot(part["threads"], part["overhead_pct"], marker="o", label=f"n = {size}")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("threads p")
    ax.set_ylabel("Spark overhead vs thread pool (%)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "framework_overhead.png", dpi=150)
    plt.close(fig)


def fig_breakdown(df: pd.DataFrame) -> None:
    """Stacked phase breakdown of the Spark family's end-to-end time."""
    spark = df[df["method"] == "blocked_ijk_distributed"]
    grouped = spark.groupby(["size", "threads"], as_index=False).agg(
        broadcast=("broadcast_ms", "median"),
        job=("job_wall_ms", "median"),
        assemble=("assemble_ms", "median"))

    sizes = sorted(grouped["size"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, size in zip(axes.flat, sizes):
        part = grouped[grouped["size"] == size].sort_values("threads")
        if part.empty:
            continue
        x = np.arange(len(part))
        ax.bar(x, part["broadcast"], label="broadcast B", color="#8e44ad")
        ax.bar(x, part["job"], bottom=part["broadcast"],
               label="job (compute + collect)", color="#c0392b")
        ax.bar(x, part["assemble"], bottom=part["broadcast"] + part["job"],
               label="driver assembly", color="#e67e22")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in part["threads"]])
        ax.set_title(f"n = {size}")
        ax.set_xlabel("threads p")
        ax.set_ylabel("median time (ms)")
    for ax in axes.flat[len(sizes):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES_DIR / "spark_phase_breakdown.png", dpi=150)
    plt.close(fig)


def fig_memory(mem: pd.DataFrame) -> None:
    rss = mem[mem["metric"] == "peak_rss"].copy()
    rss["mib"] = rss["value_bytes"] / 2**20

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("blocked_ijk_sequential", "blocked_ijk_parallel",
                   "blocked_ijk_distributed"):
        for size in sorted(rss["size"].unique()):
            part = rss[(rss["method"] == method) & (rss["size"] == size)]
            if part.empty:
                continue
            ax.plot(part["threads"], part["mib"], marker="o",
                    color=COLORS[method], alpha=0.25 + 0.15 * sorted(rss["size"].unique()).index(size))
    # one legend entry per method
    for method in ("blocked_ijk_sequential", "blocked_ijk_parallel",
                   "blocked_ijk_distributed"):
        ax.plot([], [], marker="o", color=COLORS[method], label=LABELS[method])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("threads p")
    ax.set_ylabel("peak RSS (MiB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "peak_rss_vs_threads.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    times = load_times()
    memory = load_memory()
    med = medians(times)

    fig_time(med)
    fig_speedup(med)
    fig_efficiency(med)
    fig_overhead(med)
    fig_breakdown(times)
    fig_memory(memory)
    print(f"figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
