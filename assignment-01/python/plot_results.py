"""Generate the report figures from the raw campaign CSVs.

Reads data/results/time_raw.csv and data/results/memory_raw.csv and writes
one self-contained PDF per figure into report/figures/. The figures are
reproducible from the CSVs alone.

    plot_results.py [--times PATH] [--memory PATH] [--outdir PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"python": "Python", "java": "Java", "c": "C", "go": "Go"}
COLORS = {"python": "#1f77b4", "java": "#ff7f0e", "c": "#2ca02c", "go": "#d62728"}
MARKERS = {"python": "o", "java": "s", "c": "^", "go": "D"}
LANGUAGE_ORDER = ["python", "java", "c", "go"]


def load_measured_times(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    measured = frame[frame["warmup"] == False]  # noqa: E712
    return measured[measured["batch_size"] > 0].copy()


def summarise_times(measured: pd.DataFrame) -> pd.DataFrame:
    grouped = measured.groupby(["language", "size"])["time_ns_per_multiply"]
    summary = grouped.agg(
        median_ns="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    ).reset_index()
    summary["iqr_ns"] = summary["q75"] - summary["q25"]
    summary["iqr_over_median"] = summary["iqr_ns"] / summary["median_ns"]
    return summary


def plot_median_time(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for language in LANGUAGE_ORDER:
        subset = summary[summary["language"] == language]
        axis.loglog(subset["size"], subset["median_ns"] / 1e9,
                    marker=MARKERS[language], color=COLORS[language],
                    label=LABELS[language])
    reference_n = summary["size"].max()
    reference = summary[(summary["language"] == "c")
                        & (summary["size"] == reference_n)]["median_ns"].iloc[0]
    axis.loglog(summary["size"],
                reference * (summary["size"] / reference_n) ** 3 / 1e9,
                linestyle="--", color="black", linewidth=1.2,
                label=r"$O(n^3)$ reference")
    axis.set_xlabel("matrix size $n$ (elements per side)")
    axis.set_ylabel("median time per multiplication (s)")
    axis.set_title("Runtime vs matrix size (kernel only)")
    axis.grid(True, which="both", linewidth=0.4, alpha=0.6)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "median_time_vs_size.pdf")
    plt.close(fig)


def plot_slowdown(summary: pd.DataFrame, out_dir: Path) -> None:
    pivot = summary.pivot(index="size", columns="language", values="median_ns")
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for language in LANGUAGE_ORDER:
        if language == "c":
            continue
        ratio = pivot[language] / pivot["c"]
        axis.semilogx(ratio.index, ratio, marker=MARKERS[language],
                      color=COLORS[language], label=LABELS[language])
    axis.axhline(1.0, linestyle="--", color="black", linewidth=1.0, label="C = 1")
    axis.set_xlabel("matrix size $n$")
    axis.set_ylabel("slowdown relative to C")
    axis.set_title("Runtime ratio vs the C implementation")
    axis.grid(True, which="both", linewidth=0.4, alpha=0.6)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "slowdown_vs_c.pdf")
    plt.close(fig)


def plot_peak_memory(memory: pd.DataFrame, out_dir: Path) -> None:
    peak = memory[memory["metric"] == "peak_rss"]
    theoretical = memory[memory["metric"] == "theoretical_array_storage"]
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for language in LANGUAGE_ORDER:
        subset = peak[peak["language"] == language]
        axis.loglog(subset["size"], subset["value_bytes"] / 1e6,
                    marker=MARKERS[language], color=COLORS[language],
                    label=LABELS[language])
    ref = theoretical[theoretical["language"] == "c"]
    axis.loglog(ref["size"], ref["theoretical_bytes"] / 1e6,
                linestyle="--", color="black", linewidth=1.2,
                label=r"theoretical arrays $24n^2$")
    axis.set_xlabel("matrix size $n$ (elements per side)")
    axis.set_ylabel("peak RSS (MB)")
    axis.set_title("Peak resident memory vs matrix size")
    axis.grid(True, which="both", linewidth=0.4, alpha=0.6)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "peak_rss_vs_size.pdf")
    plt.close(fig)


def plot_variability(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for language in LANGUAGE_ORDER:
        subset = summary[summary["language"] == language]
        axis.semilogx(subset["size"], 100 * subset["iqr_over_median"],
                      marker=MARKERS[language], color=COLORS[language],
                      label=LABELS[language])
    axis.set_xlabel("matrix size $n$")
    axis.set_ylabel("IQR / median (%)")
    axis.set_title("Relative timing variability (IQR of 7 repetitions)")
    axis.grid(True, which="both", linewidth=0.4, alpha=0.6)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "timing_variability.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", type=Path,
                        default=PROJECT_ROOT / "data" / "results" / "time_raw.csv")
    parser.add_argument("--memory", type=Path,
                        default=PROJECT_ROOT / "data" / "results" / "memory_raw.csv")
    parser.add_argument("--outdir", type=Path,
                        default=PROJECT_ROOT / "report" / "figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    measured = load_measured_times(args.times)
    summary = summarise_times(measured)
    memory = pd.read_csv(args.memory)
    plot_median_time(summary, args.outdir)
    plot_slowdown(summary, args.outdir)
    plot_peak_memory(memory, args.outdir)
    plot_variability(summary, args.outdir)
    print(f"figures written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
