"""Plot the assignment 2 campaign: dense optimisation (Part A) and sparse CSR (Part B).

Every figure is regenerated from the raw CSVs alone and written as PDF into
report/figures/ for inclusion in the LaTeX report:

- median_time_vs_size.pdf       Part A: kernel time per method vs n (log-log)
- speedup_vs_baseline.pdf       Part A: speedup of the optimised methods
- sparse_time_vs_density.pdf    Part B: CSR vs matched dense, time vs density
- sparse_speedup_vs_density.pdf Part B: dense / CSR speedup vs density
- nnz_growth_vs_density.pdf     Part B: product densification (nnz_c / n^2)
- conversion_vs_kernel.pdf      Part B: dense-to-CSR conversion vs kernel time
- peak_rss_all.pdf              memory: peak RSS for both parts
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

DENSE_METHOD_LABELS = {
    "triple_loop_ijk_dense": "baseline i-j-k",
    "triple_loop_ikj_dense": "i-k-j reordering",
    "blocked_ijk_dense": "cache blocking (64)",
}
BASELINE_METHOD = "triple_loop_ijk_dense"
CSR_METHOD = "sparse_csr_gustavson"


def measured(df: pd.DataFrame) -> pd.DataFrame:
    """Keep measured (non warm-up) samples only.

    pandas may parse the warmup column either as bool or as string, so the
    filter accepts both spellings.
    """
    warmup = df["warmup"]
    keep = warmup.map(lambda value: value in (False, "false"))
    return df[keep].copy()


def median_iqr(df: pd.DataFrame, group: list[str],
               column: str = "time_ns_per_multiply") -> pd.DataFrame:
    """Median and interquartile range of the given time column per group."""
    stats = df.groupby(group)[column].agg(
        median="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
    ).reset_index()
    stats["iqr"] = stats["q3"] - stats["q1"]
    return stats


def plot_dense_time_vs_size(dense: pd.DataFrame) -> None:
    stats = median_iqr(dense, ["size", "method"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method, label in DENSE_METHOD_LABELS.items():
        rows = stats[stats["method"] == method].sort_values("size")
        ax.plot(rows["size"], rows["median"] / 1e9, "o-", label=label)
        ax.fill_between(rows["size"], rows["q1"] / 1e9, rows["q3"] / 1e9, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("kernel time per multiplication (s, median with IQR)")
    ax.set_title("Dense methods: runtime vs matrix size")
    ax.legend()
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "median_time_vs_size.pdf")
    plt.close(fig)


def plot_dense_speedup(dense: pd.DataFrame) -> None:
    stats = median_iqr(dense, ["size", "method"])
    baseline = stats[stats["method"] == BASELINE_METHOD][["size", "median"]]
    baseline = baseline.rename(columns={"median": "baseline_ns"})
    merged = stats.merge(baseline, on="size")
    merged["speedup"] = merged["baseline_ns"] / merged["median"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, label in DENSE_METHOD_LABELS.items():
        if method == BASELINE_METHOD:
            continue
        rows = merged[merged["method"] == method].sort_values("size")
        ax.plot(rows["size"], rows["speedup"], "o-", label=label)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("speedup vs baseline i-j-k")
    ax.set_title("Optimised dense methods: speedup over the baseline")
    ax.legend()
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "speedup_vs_baseline.pdf")
    plt.close(fig)


def plot_sparse_time_vs_density(sparse: pd.DataFrame) -> None:
    stats = median_iqr(sparse, ["size", "density", "representation"],
                  column="kernel_time_ns_per_multiply")
    sizes = sorted(stats["size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.1 * len(sizes), 3.6),
                             sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    for ax, size in zip(axes, sizes):
        rows = stats[stats["size"] == size]
        for representation, label in (("csr", "CSR (Gustavson)"),
                                      ("dense", "dense i-k-j")):
            part = rows[rows["representation"] == representation].sort_values("density")
            if part.empty:
                continue
            ax.plot(part["density"], part["median"] / 1e9, "o-", label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("density (%)")
        ax.set_title(f"n = {size}")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel("kernel time per multiplication (s, median)")
    axes[0].legend()
    fig.suptitle("Sparse CSR vs matched dense comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sparse_time_vs_density.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sparse_speedup_vs_density(sparse: pd.DataFrame) -> None:
    stats = median_iqr(sparse, ["size", "density", "representation"],
                  column="kernel_time_ns_per_multiply")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for size in sorted(stats["size"].unique()):
        rows = stats[stats["size"] == size]
        csr = rows[rows["representation"] == "csr"][["density", "median"]]
        dense = rows[rows["representation"] == "dense"][["density", "median"]]
        merged = csr.merge(dense, on="density", suffixes=("_csr", "_dense"))
        if merged.empty:
            continue
        merged["speedup"] = merged["median_dense"] / merged["median_csr"]
        merged = merged.sort_values("density")
        ax.plot(merged["density"], merged["speedup"], "o-", label=f"n = {size}")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("density (%)")
    ax.set_ylabel("speedup: dense i-k-j / CSR")
    ax.set_title("Where does the sparse representation win?")
    ax.legend()
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sparse_speedup_vs_density.pdf")
    plt.close(fig)


def plot_nnz_growth(sparse: pd.DataFrame) -> None:
    csr_rows = sparse[sparse["representation"] == "csr"].copy()
    summary = csr_rows.groupby(["size", "density"])[
        ["nnz_a", "nnz_b", "nnz_c"]].median().reset_index()
    summary["output_fraction"] = summary["nnz_c"] / (summary["size"] ** 2)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for size in sorted(summary["size"].unique()):
        rows = summary[summary["size"] == size].sort_values("density")
        ax.plot(rows["density"], rows["output_fraction"], "o-", label=f"n = {size}")
    ax.set_xscale("log")
    ax.set_xlabel("input density (%)")
    ax.set_ylabel("output density nnz(C) / n²")
    ax.set_title("The product densifies: output nonzeros vs input density")
    ax.legend()
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "nnz_growth_vs_density.pdf")
    plt.close(fig)


def plot_conversion_vs_kernel(sparse: pd.DataFrame) -> None:
    csr_rows = sparse[sparse["representation"] == "csr"].copy()
    conversion = csr_rows.groupby(["size", "density"])["conversion_time_ns"].median().reset_index()
    kernel = median_iqr(csr_rows, ["size", "density"],
                        column="kernel_time_ns_per_multiply")
    merged = conversion.merge(kernel, on=["size", "density"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for size in sorted(merged["size"].unique()):
        rows = merged[merged["size"] == size].sort_values("density")
        ax.plot(rows["density"], rows["conversion_time_ns"] / 1e6, "s--",
                label=f"conversion n = {size}")
        ax.plot(rows["density"], rows["median"] / 1e6, "o-",
                label=f"kernel n = {size}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("density (%)")
    ax.set_ylabel("time (ms, median)")
    ax.set_title("Dense-to-CSR conversion vs CSR kernel time")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "conversion_vs_kernel.pdf")
    plt.close(fig)


def plot_peak_rss(dense: pd.DataFrame, memory: pd.DataFrame) -> None:
    fig, (ax_dense, ax_sparse) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    part_a = memory[(memory["density"].isna()) & (memory["metric"] == "peak_rss")]
    for method, label in DENSE_METHOD_LABELS.items():
        rows = part_a[part_a["method"] == method].sort_values("size")
        ax_dense.plot(rows["size"], rows["value_bytes"] / 2**20, "o-", label=label)
    ax_dense.set_xscale("log")
    ax_dense.set_xlabel("matrix size n")
    ax_dense.set_ylabel("peak RSS (MiB)")
    ax_dense.set_title("Part A: peak resident memory")
    ax_dense.legend(fontsize=8)
    ax_dense.grid(True, which="both", linewidth=0.4, alpha=0.6)

    part_b = memory[(memory["density"].notna()) & (memory["metric"] == "peak_rss")]
    csr = part_b[part_b["method"] == CSR_METHOD]
    for size in sorted(csr["size"].unique()):
        rows = csr[csr["size"] == size].sort_values("density")
        ax_sparse.plot(rows["density"], rows["value_bytes"] / 2**20, "o-",
                       label=f"CSR n = {size}")
    ax_sparse.set_xscale("log")
    ax_sparse.set_xlabel("density (%)")
    ax_sparse.set_ylabel("peak RSS (MiB)")
    ax_sparse.set_title("Part B: peak resident memory (CSR)")
    ax_sparse.legend(fontsize=8)
    ax_sparse.grid(True, which="both", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "peak_rss_all.pdf")
    plt.close(fig)


def main() -> int:
    dense_path = RESULTS_DIR / "dense_time_raw.csv"
    sparse_path = RESULTS_DIR / "sparse_time_raw.csv"
    memory_path = RESULTS_DIR / "memory_raw.csv"
    for path in (dense_path, sparse_path, memory_path):
        if not path.is_file():
            print(f"missing results file: {path}", file=sys.stderr)
            return 1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dense = measured(pd.read_csv(dense_path))
    sparse = measured(pd.read_csv(sparse_path))
    memory = pd.read_csv(memory_path)

    plot_dense_time_vs_size(dense)
    plot_dense_speedup(dense)
    plot_sparse_time_vs_density(sparse)
    plot_sparse_speedup_vs_density(sparse)
    plot_nnz_growth(sparse)
    plot_conversion_vs_kernel(sparse)
    plot_peak_rss(dense, memory)
    print(f"figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
