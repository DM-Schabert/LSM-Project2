"""IO1 (Whiteboard "Steps, not stars"): how much of a run is IO?

IO here is everything that moves the stars in or out of the program:
load_time_s (rank 0 reads the CSV) + distribute_time_s (the Bcast to every
rank) + output_time_s (the final gather and writing --save-final). It
happens once per run, so its share of the run shrinks as the run gets longer.

Two figures from results.csv, every run in it (serial, 1-node MPI, 2-node MPI):
    io_share_vs_steps   IO as % of the whole run, against steps per run
    io_time_vs_stars    IO time per run in seconds, against N
and a short summary table on the screen.

Plots have no titles (project convention); the file name says what they show.
Serial is orange and MPI blue, as in plot_results.py; 2-node runs are triangles.

Usage (from the project root):
    python3 IO_Improvements/PlotIOShare.py
    python3 IO_Improvements/PlotIOShare.py --results results.csv --plots-dir Plots
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # files only, works on the HPC too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# The palette and style of plot_results.py
COLOUR_SERIAL: str = "#eb6834"
COLOUR_MPI: str = "#2a78d6"
INK: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
GRID: str = "#e4e3df"
GUIDE: str = "#9a9893"
DASH: Tuple[int, Tuple[int, int]] = (0, (4, 3))

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

# (label, colour, marker) per group, always in this order
GROUPS: List[Tuple[str, str, str]] = [
    ("Serial", COLOUR_SERIAL, "o"),
    ("MPI, 1 node", COLOUR_MPI, "o"),
    ("MPI, 2 nodes", COLOUR_MPI, "^"),
]
IO_COLUMNS: List[str] = ["load_time_s", "distribute_time_s", "output_time_s"]


def thousands(x: float, _pos: int | None = None) -> str:
    """1000 -> '1.000' (dots as thousands separators, as in the simulation output)."""
    return f"{x:,.0f}".replace(",", ".") if x >= 1 else f"{x:g}"


def load_runs(path: Path) -> pd.DataFrame:
    runs: pd.DataFrame = pd.read_csv(path, low_memory=False)
    runs = runs[(runs["total_time_s"] > 0) & (runs["iterations"] > 0)].copy()
    runs["io_s"] = runs[[c for c in IO_COLUMNS if c in runs]].fillna(0).sum(axis=1)
    runs["io_share_pct"] = runs["io_s"] / runs["total_time_s"] * 100
    nodes: pd.Series = runs["n_nodes"].fillna(1) if "n_nodes" in runs else pd.Series(1, index=runs.index)
    runs["group"] = np.where(runs["implementation"] == "serial", "Serial",
                             np.where(nodes >= 2, "MPI, 2 nodes", "MPI, 1 node"))
    return runs


def new_axes() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def scatter_groups(ax: Axes, runs: pd.DataFrame, x: str, y: str, legend_loc: str) -> None:
    for label, colour, marker in GROUPS:
        part: pd.DataFrame = runs[runs["group"] == label]
        if part.empty:
            continue
        ax.scatter(part[x], part[y], s=22, marker=marker, color=colour, alpha=0.55,
                   edgecolors="white", linewidths=0.5, label=f"{label} ({thousands(len(part))} runs)")
    ax.legend(loc=legend_loc, fontsize=9, labelcolor=INK_SECONDARY)


def log_axes(ax: Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_major_formatter(FuncFormatter(thousands))
        axis.set_minor_formatter(NullFormatter())


def save(fig: Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved {folder / name}.png and .pdf")


def plot_share_vs_steps(runs: pd.DataFrame, folder: Path) -> None:
    fig, ax = new_axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    scatter_groups(ax, runs, "iterations", "io_share_pct", legend_loc="lower left")
    ax.axhline(1.0, color=GUIDE, linestyle=DASH, linewidth=1.0)
    ax.text(40, 1.1, "1 % of the run", color=INK_SECONDARY, fontsize=8.5, va="bottom")
    log_axes(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g} %"))
    ax.set_xlabel("Steps in the run")
    ax.set_ylabel("IO (load + distribute + output), % of the run")
    save(fig, folder, "io_share_vs_steps")


def plot_time_vs_stars(runs: pd.DataFrame, folder: Path) -> None:
    fig, ax = new_axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    scatter_groups(ax, runs, "n_stars_start", "io_s", legend_loc="lower right")
    ax.axhline(1.0, color=GUIDE, linestyle=DASH, linewidth=1.0)
    ax.text(ax.get_xlim()[0] * 1.15, 1.1, "1 s = 0.3 % of a 300 s run", color=INK_SECONDARY,
            fontsize=8.5, va="bottom")
    ax.set_ylim(top=max(2.0, float(runs["io_s"].max()) * 1.5))
    log_axes(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g} s"))
    ax.set_xlabel("Number of stars N")
    ax.set_ylabel("IO time per run")
    save(fig, folder, "io_time_vs_stars")


def summary(runs: pd.DataFrame) -> None:
    table: pd.DataFrame = runs.groupby("group").agg(
        runs=("io_s", "size"), io_median_s=("io_s", "median"), io_max_s=("io_s", "max"),
        share_median_pct=("io_share_pct", "median"), share_max_pct=("io_share_pct", "max"))
    print(table.reindex([g[0] for g in GROUPS]).dropna(how="all").round(3).to_string())
    long_runs: pd.DataFrame = runs[runs["iterations"] >= 1000]
    print(f"Runs with at least 1.000 steps: {len(long_runs)}, IO share median "
          f"{long_runs['io_share_pct'].median():.2f} %, max {long_runs['io_share_pct'].max():.2f} %")


def main() -> None:
    parser = argparse.ArgumentParser(description="IO share of every run in results.csv (IO1).")
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--plots-dir", type=Path, default=Path("Plots"))
    args = parser.parse_args()
    runs: pd.DataFrame = load_runs(args.results)
    summary(runs)
    plot_share_vs_steps(runs, args.plots_dir)
    plot_time_vs_stars(runs, args.plots_dir)


if __name__ == "__main__":
    main()
