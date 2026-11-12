"""Plots for the report, made from results.csv.

Reads every run in results.csv and saves each figure as PNG (viewing) and PDF (LaTeX):

    step_time_vs_N             time per step against number of stars (log-log): serial,
                               each MPI rank count, an N² guide, and a fit of the
                               serial cost model  T(N) = a·N + b·N²
    speedup_vs_N               speed-up of each MPI rank count against N, with the
                               ideal (= number of ranks) as a dashed line
    interactions_per_core_vs_N pair interactions per second per core: how well the
                               NumPy kernel uses a core, serial vs. MPI
    overhead_share_vs_N        share of an average MPI rank's time spent communicating
                               (Allgatherv) and waiting (sync), against N
    serial_vs_mpi_per_node     serial vs. MPI time per step for the same N on
                               different nodes (only drawn for N run on 2+ nodes)
    speedup_vs_ranks           speed-up against number of ranks, with the ideal line
                               (only drawn once runs exist at 2+ rank counts: E2)
    weak_scaling_efficiency    weak scaling (Gustafson): efficiency E(P) = T(1)/T(P)
                               for runs where N grows as N0·√P (constant work per
                               rank for O(N²) direct summation); ideal is 1
    gustafson_scaled_speedup   Gustafson's scaled speed-up S(P) = τ(P)/T(P), with
                               the ideal S = P and a fit S = s + p·P that gives the
                               serial fraction s

Weak-scaling runs are found automatically: a series is a run at P = 1 with N0
stars plus runs at two or more other rank counts with N within ±3 % of N0·√P.
τ(P), the time ONE core would need for the scaled problem, comes from the measured
T(1) and the serial cost model: τ(P) = T(1) · model(N(P)) / model(N0), with
model(N) = a·N + b·N². Without a model fit it falls back to τ(P) = T(1)·P (pure N²).

Rules for the numbers:
    * Everything is compared per time step (step_time_mean_ms). Runs that hit the
      time limit (completed=False) are still valid for that, so they are used
      unless --completed-only is given. Runs with fewer than --min-steps steps
      are dropped (too few steps for a stable mean).
    * Speed-up = serial time per step / MPI time per step, only between runs on
      the SAME node (hostname) with the same data, dt, softening and merge
      radius. Nodes with the same CPU model differed by ~16 % in the first runs,
      so a cross-node speed-up would mix node speed into the result.

Plots have no titles (project convention); the file name says what they show.

Usage:
    python3 plot_results.py                        # results.csv -> Plots/
    python3 plot_results.py --results my.csv --plots-dir figs --completed-only
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

# Validated reference palette (light mode). Serial is always orange; MPI rank
# counts take the remaining slots in a fixed order, so a rank count keeps its
# colour in every figure.
COLOUR_SERIAL: str = "#eb6834"       # slot 2
MPI_COLOURS: List[str] = ["#2a78d6",  # slot 1
                          "#1baf7a",  # slot 3
                          "#6250d6",  # slot 7
                          "#e87ba4"]  # slot 5
# Overhead metrics get their own hues, so they are never mistaken for "serial"
# (orange) or "MPI" (blue) from the other figures.
COLOUR_COMM: str = "#1baf7a"         # communication (slot 3)
COLOUR_WAIT: str = "#6250d6"         # waiting / sync (slot 7)
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

PAIR_KEYS: List[str] = ["hostname", "data_file", "dt", "softening_pc", "merge_radius_pc"]


# ---------------------------------------------------------------- helpers

def thousands(x: float, _pos: int | None = None) -> str:
    """1000 -> '1.000' (dots as thousands separators, as in the simulation output)."""
    return f"{x:,.0f}".replace(",", ".")


def nice_ticks(lo: float, hi: float) -> List[float]:
    """1-2-3-5 ticks between lo and hi, for log axes."""
    return [t for t in (10.0 ** k * m for k in range(-3, 8) for m in (1, 2, 3, 5)) if lo <= t <= hi]


def log_axis(axis, lo: float, hi: float, fmt=thousands) -> None:
    axis.set_major_locator(FixedLocator(nice_ticks(lo, hi)))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_formatter(NullFormatter())


def number_formatter(x: float, _pos: int | None = None) -> str:
    """Thousands for big numbers, plain decimals for small ones (log axes below 1)."""
    return thousands(x) if x >= 1 else f"{x:g}"


def save(fig: Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved {folder / name}.png and .pdf")


def new_axes(width: float = 6.0, height: float = 4.2) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def mpi_rank_counts(runs: pd.DataFrame) -> List[int]:
    return sorted(int(r) for r in runs.loc[runs["implementation"] == "mpi", "n_ranks"].unique())


def mpi_colour(ranks: int, all_ranks: List[int]) -> str:
    return MPI_COLOURS[min(all_ranks.index(ranks), len(MPI_COLOURS) - 1)]


def n_axis(ax: Axes, runs: pd.DataFrame, right_room: float = 1.6) -> None:
    """Log x-axis for N, ticked at the N values that were run."""
    n_values: np.ndarray = np.sort(runs["n_stars_start"].unique())
    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(n_values.min() * 0.75, n_values.max() * right_room)
    ax.set_xlabel("Number of stars N")
    # rotate when two neighbouring ticks are close on the log axis (e.g. 10.000 and 15.000)
    if len(n_values) > 1 and np.min(np.diff(np.log10(n_values))) < 0.25:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def end_label(ax: Axes, x: float, y: float, text: str) -> None:
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points",
                va="center", ha="left", color=INK, fontsize=9)


# ---------------------------------------------------------------- data

def load_runs(path: Path, completed_only: bool, min_steps: int) -> pd.DataFrame:
    runs: pd.DataFrame = pd.read_csv(path)
    keep: pd.Series = runs["iterations"] >= min_steps
    if completed_only:
        keep &= runs["completed"].astype(str) == "True"
    dropped: pd.DataFrame = runs[~keep]
    for _, r in dropped.iterrows():
        print(f"Skipping {r['run_name']} ({int(r['iterations'])} steps, completed={r['completed']})")
    early: pd.DataFrame = runs[keep & (runs["completed"].astype(str) != "True")]
    for _, r in early.iterrows():
        print(f"Using {r['run_name']} although it stopped early ({r['stop_reason']}, "
              f"{int(r['iterations'])} of {int(r['steps_requested'])} steps): time per step is still valid")
    used: pd.DataFrame = runs[keep].copy()
    used["step_ms"] = used["step_time_mean_ms"]
    used["n_ranks"] = used["n_ranks"].astype(int)
    used["per_core"] = used["interactions_per_s"] / used["n_ranks"]
    return used


def matched_pairs(runs: pd.DataFrame) -> pd.DataFrame:
    """Every MPI run with the median serial time per step on the same node and settings."""
    serial: pd.DataFrame = runs[runs["implementation"] == "serial"]
    mpi: pd.DataFrame = runs[runs["implementation"] == "mpi"]
    serial_med: pd.DataFrame = (serial.groupby(PAIR_KEYS, as_index=False)["step_ms"].median()
                                .rename(columns={"step_ms": "serial_step_ms"}))
    pairs: pd.DataFrame = mpi.merge(serial_med, on=PAIR_KEYS)
    pairs["speedup"] = pairs["serial_step_ms"] / pairs["step_ms"]
    pairs["efficiency"] = pairs["speedup"] / pairs["n_ranks"]
    return pairs


def fit_serial_model(runs: pd.DataFrame) -> Tuple[float, float] | None:
    """Least-squares fit of the serial time per step: T(N) = a·N + b·N² (ms).

    a = cost per row of the Python loop over i (overhead, independent of N),
    b = cost per pair interaction. Needs serial runs at 3+ different N.
    """
    serial: pd.Series = (runs[runs["implementation"] == "serial"]
                         .groupby("n_stars_start")["step_ms"].median())
    if len(serial) < 3:
        return None
    n: np.ndarray = serial.index.to_numpy(dtype=float)
    t: np.ndarray = serial.to_numpy(dtype=float)
    # weight by 1/t so small N counts as much as large N (relative error)
    design: np.ndarray = np.column_stack([n, n ** 2]) / t[:, None]
    (a, b), *_ = np.linalg.lstsq(design, np.ones_like(t), rcond=None)
    return float(a), float(b)


# ---------------------------------------------------------------- figures

def plot_step_time_vs_n(runs: pd.DataFrame, fit: Tuple[float, float] | None, folder: Path) -> None:
    fig, ax = new_axes()
    all_ranks: List[int] = mpi_rank_counts(runs)
    series: List[Tuple[str, pd.DataFrame, str]] = []
    serial: pd.DataFrame = runs[runs["implementation"] == "serial"]
    if not serial.empty:
        series.append(("Serial", serial, COLOUR_SERIAL))
    for r in all_ranks:
        series.append((f"MPI, {r} ranks", runs[(runs["implementation"] == "mpi") & (runs["n_ranks"] == r)],
                       mpi_colour(r, all_ranks)))

    for name, rows, colour in series:
        med: pd.Series = rows.groupby("n_stars_start")["step_ms"].median().sort_index()
        ax.plot(med.index, med.values, color=colour, linewidth=2, zorder=3)
        ax.scatter(rows["n_stars_start"], rows["step_ms"], s=36, color=colour, edgecolors="white",
                   linewidths=1.5, zorder=4, label=name)
        end_label(ax, med.index[-1], med.values[-1], name)

    n_line: np.ndarray = np.geomspace(runs["n_stars_start"].min() * 0.85, runs["n_stars_start"].max() * 1.15, 80)
    if fit is not None:
        a, b = fit
        ax.plot(n_line, a * n_line + b * n_line ** 2, color=COLOUR_SERIAL, linewidth=1, linestyle=DASH,
                zorder=2, label=r"serial fit $aN + bN^2$")
    # N² guide through the largest-N point of the largest MPI series
    ref_rows: pd.DataFrame = series[-1][1]
    ref: pd.Series = ref_rows.groupby("n_stars_start")["step_ms"].median().sort_index()
    ax.plot(n_line, ref.values[-1] * (n_line / ref.index[-1]) ** 2, color=GUIDE, linewidth=1.2,
            linestyle=DASH, zorder=1, label=r"$\propto N^2$ (guide)")

    ax.set_yscale("log")
    n_axis(ax, runs)
    ax.set_ylim(runs["step_ms"].min() * 0.6, runs["step_ms"].max() * 1.6)
    log_axis(ax.yaxis, *ax.get_ylim(), fmt=number_formatter)
    ax.set_ylabel("Time per step [ms]")
    ax.legend(loc="upper left")
    save(fig, folder, "step_time_vs_N")


def plot_speedup_vs_n(pairs: pd.DataFrame, runs: pd.DataFrame, folder: Path) -> None:
    if pairs.empty:
        print("No serial+MPI pairs on the same node; skipping speedup_vs_N")
        return
    fig, ax = new_axes()
    all_ranks: List[int] = mpi_rank_counts(runs)
    for r in sorted(pairs["n_ranks"].unique()):
        rows: pd.DataFrame = pairs[pairs["n_ranks"] == r]
        colour: str = mpi_colour(int(r), all_ranks)
        med: pd.Series = rows.groupby("n_stars_start")["speedup"].median().sort_index()
        ax.axhline(r, color=GUIDE, linewidth=1.2, linestyle=DASH, zorder=1)
        ax.annotate(f"ideal ({r} ranks)", (pairs["n_stars_start"].min() * 0.78, r), xytext=(0, 4),
                    textcoords="offset points", fontsize=8.5, color=INK_SECONDARY, va="bottom")
        ax.plot(med.index, med.values, color=colour, linewidth=2, zorder=3)
        ax.scatter(rows["n_stars_start"], rows["speedup"], s=36, color=colour, edgecolors="white",
                   linewidths=1.5, zorder=4, label=f"MPI, {r} ranks")
        # values below the points, alternating above/below where neighbours are close
        prev_n: float | None = None
        below: bool = True
        for n, s in med.items():
            close: bool = prev_n is not None and np.log10(n / prev_n) < 0.15
            below = (not below) if close else True
            ax.annotate(f"{s:.2f}", (n, s), xytext=(0, -14 if below else 8), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_SECONDARY)
            prev_n = n
    n_axis(ax, pairs, right_room=1.3)
    ax.set_ylim(0, pairs["n_ranks"].max() * 1.15)
    ax.set_ylabel(r"Speed-up  $T_\mathrm{serial} / T_\mathrm{MPI}$ per step")
    ax.legend(loc="lower right")
    save(fig, folder, "speedup_vs_N")


def plot_interactions_per_core(runs: pd.DataFrame, folder: Path) -> None:
    fig, ax = new_axes()
    all_ranks: List[int] = mpi_rank_counts(runs)
    groups: List[Tuple[str, pd.DataFrame, str]] = []
    serial: pd.DataFrame = runs[runs["implementation"] == "serial"]
    if not serial.empty:
        groups.append(("Serial", serial, COLOUR_SERIAL))
    for r in all_ranks:
        groups.append((f"MPI, {r} ranks (per rank)",
                       runs[(runs["implementation"] == "mpi") & (runs["n_ranks"] == r)], mpi_colour(r, all_ranks)))
    for name, rows, colour in groups:
        med: pd.Series = (rows.groupby("n_stars_start")["per_core"].median() / 1e6).sort_index()
        ax.plot(med.index, med.values, color=colour, linewidth=2, zorder=3)
        ax.scatter(rows["n_stars_start"], rows["per_core"] / 1e6, s=36, color=colour, edgecolors="white",
                   linewidths=1.5, zorder=4, label=name)
    n_axis(ax, runs, right_room=1.3)
    ax.set_ylim(0, (runs["per_core"].max() / 1e6) * 1.2)
    ax.set_ylabel("Pair interactions per second per core [millions]")
    ax.legend(loc="lower right")
    save(fig, folder, "interactions_per_core_vs_N")


def plot_overhead_share(runs: pd.DataFrame, folder: Path) -> None:
    """Share of an average rank's time spent in communication and in waiting (sync)."""
    mpi: pd.DataFrame = runs[runs["implementation"] == "mpi"].copy()
    if mpi.empty:
        return
    ranks: int = int(mpi["n_ranks"].mode().iloc[0])  # the most common rank count
    mpi = mpi[mpi["n_ranks"] == ranks]
    parts: List[str] = ["forces_time_mean_s", "communication_time_mean_s", "sync_time_mean_s",
                        "integrate_time_mean_s", "merge_time_mean_s"]
    total: pd.Series = mpi[parts].fillna(0).sum(axis=1)
    mpi["comm_pct"] = 100 * mpi["communication_time_mean_s"] / total
    mpi["wait_pct"] = 100 * mpi["sync_time_mean_s"] / total

    fig, ax = new_axes()
    for col, name, colour in (("wait_pct", "Waiting at collectives (sync)", COLOUR_WAIT),
                              ("comm_pct", "Communication (Allgatherv)", COLOUR_COMM)):
        med: pd.Series = mpi.groupby("n_stars_start")[col].median().sort_index()
        ax.plot(med.index, med.values, color=colour, linewidth=2, zorder=3)
        ax.scatter(mpi["n_stars_start"], mpi[col], s=36, color=colour, edgecolors="white",
                   linewidths=1.5, zorder=4, label=name)
    ax.set_yscale("log")
    n_axis(ax, mpi, right_room=1.3)
    lo: float = float(mpi[["comm_pct", "wait_pct"]].min().min()) * 0.5
    hi: float = float(mpi[["comm_pct", "wait_pct"]].max().max()) * 2
    ax.set_ylim(lo, hi)
    log_axis(ax.yaxis, lo, hi, fmt=lambda x, _p=None: f"{x:g} %")
    ax.set_ylabel(f"Share of an average rank's time ({ranks} ranks)")
    ax.legend(loc="upper right")
    save(fig, folder, "overhead_share_vs_N")


def plot_serial_vs_mpi_per_node(pairs: pd.DataFrame, folder: Path) -> None:
    """Grouped bars for every N that was run (serial + MPI) on two or more nodes."""
    if pairs.empty:
        return
    multi: pd.Series = pairs.groupby(["n_stars_start", "n_ranks"])["hostname"].nunique()
    multi = multi[multi >= 2]
    if multi.empty:
        print("No N with serial+MPI pairs on 2+ nodes; skipping serial_vs_mpi_per_node")
        return
    n_sel, r_sel = multi.index[-1]  # the largest such N
    sel: pd.DataFrame = pairs[(pairs["n_stars_start"] == n_sel) & (pairs["n_ranks"] == r_sel)]
    hosts: List[str] = sorted(sel["hostname"].unique())
    serial_ms: List[float] = [float(sel.loc[sel["hostname"] == h, "serial_step_ms"].median()) for h in hosts]
    mpi_ms: List[float] = [float(sel.loc[sel["hostname"] == h, "step_ms"].median()) for h in hosts]
    speedups: List[float] = [s / m for s, m in zip(serial_ms, mpi_ms)]

    fig, ax = new_axes(1.9 * len(hosts) + 2.2, 4.2)
    ax.grid(False, axis="x")
    x: np.ndarray = np.arange(len(hosts))
    width, gap = 0.36, 0.02
    ax.bar(x - width / 2 - gap / 2, serial_ms, width, color=COLOUR_SERIAL, label="Serial", zorder=3)
    ax.bar(x + width / 2 + gap / 2, mpi_ms, width, color=MPI_COLOURS[0], label=f"MPI, {r_sel} ranks", zorder=3)
    for xi, s, m, sp in zip(x, serial_ms, mpi_ms, speedups):
        for xpos, val in ((xi - width / 2 - gap / 2, s), (xi + width / 2 + gap / 2, m)):
            ax.annotate(thousands(val), (xpos, val), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        ax.annotate(f"speed-up {sp:.2f}×", (xi, s), xytext=(0, 18), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(hosts)
    ax.set_xlabel(f"Node (N = {thousands(n_sel)})")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylabel("Time per step [ms]")
    ax.set_ylim(0, max(serial_ms) * 1.22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=2)
    save(fig, folder, "serial_vs_mpi_per_node")


def plot_speedup_vs_ranks(pairs: pd.DataFrame, folder: Path) -> None:
    if pairs.empty or pairs["n_ranks"].nunique() < 2:
        print("Need serial+MPI pairs at 2+ rank counts for speedup_vs_ranks (the E2 scaling runs); "
              "skipping it for now")
        return
    fig, ax = new_axes()
    p_max: int = int(pairs["n_ranks"].max())
    ax.plot([1, p_max], [1, p_max], color=GUIDE, linewidth=1.2, linestyle=DASH, label="Ideal")
    for i, (n, rows) in enumerate(sorted(pairs.groupby("n_stars_start"))):
        colour: str = MPI_COLOURS[min(i, len(MPI_COLOURS) - 1)]
        med: pd.Series = rows.groupby("n_ranks")["speedup"].median().sort_index()
        ax.plot(med.index, med.values, color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=f"N = {thousands(n)}")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel(r"Speed-up  $T_\mathrm{serial} / T_P$")
    ax.set_xlim(0, p_max * 1.05)
    ax.set_ylim(0, p_max * 1.05)
    ax.legend(loc="upper left")
    save(fig, folder, "speedup_vs_ranks")


# ---------------------------------------------------------------- weak scaling (Gustafson)

WEAK_TOLERANCE: float = 0.03  # N may differ from N0·√P by this much (subset sizes are rounded)


def weak_scaling_series(runs: pd.DataFrame, fit: Tuple[float, float] | None) -> List[pd.DataFrame]:
    """Find weak-scaling series and compute T(P), E(P) and S(P) for each.

    A series is a P = 1 run with N0 stars (MPI with 1 rank preferred, else serial)
    plus MPI runs at 2+ other rank counts with N ≈ N0·√P and the same dt,
    softening and merge radius. Returns one table per series, one row per P
    (medians over repeats).
    """
    candidates: pd.DataFrame = runs.copy()
    candidates["n0_est"] = candidates["n_stars_start"] / np.sqrt(candidates["n_ranks"])
    settings: List[str] = ["dt", "softening_pc", "merge_radius_pc"]
    found: List[pd.DataFrame] = []
    bases: pd.DataFrame = candidates[candidates["n_ranks"] == 1]

    for (n0, *setting_values), _ in bases.groupby(["n_stars_start"] + settings):
        same: pd.Series = pd.Series(True, index=candidates.index)
        for col, val in zip(settings, setting_values):
            same &= candidates[col] == val
        near: pd.DataFrame = candidates[same & (np.abs(candidates["n0_est"] / n0 - 1) <= WEAK_TOLERANCE)]
        mpi_scaled: pd.DataFrame = near[(near["implementation"] == "mpi") & (near["n_ranks"] > 1)]
        if mpi_scaled["n_ranks"].nunique() < 2:
            continue

        # T(1): an MPI run on 1 rank uses the same code path; otherwise the serial run
        base_rows: pd.DataFrame = near[(near["n_ranks"] == 1) & (near["n_stars_start"] == n0)]
        mpi_base: pd.DataFrame = base_rows[base_rows["implementation"] == "mpi"]
        base: pd.DataFrame = mpi_base if not mpi_base.empty else base_rows
        rows: List[dict] = [{"P": 1, "N": int(n0), "T_ms": float(base["step_ms"].median()),
                             "repeats": len(base), "hosts": ",".join(sorted(base["hostname"].unique()))}]
        for p, grp in mpi_scaled.groupby("n_ranks"):
            rows.append({"P": int(p), "N": int(grp["n_stars_start"].median()),
                         "T_ms": float(grp["step_ms"].median()), "repeats": len(grp),
                         "hosts": ",".join(sorted(grp["hostname"].unique()))})
        table: pd.DataFrame = pd.DataFrame(rows).sort_values("P").reset_index(drop=True)

        t1: float = table.loc[0, "T_ms"]
        table["E"] = t1 / table["T_ms"]
        if fit is not None:
            a, b = fit
            model = lambda n: a * n + b * n ** 2  # noqa: E731 - small local helper
            table["tau_ms"] = t1 * model(table["N"].astype(float)) / model(float(n0))
        else:
            table["tau_ms"] = t1 * table["P"]
        table["S"] = table["tau_ms"] / table["T_ms"]
        table.attrs["n0"] = int(n0)
        table.attrs["tau_source"] = "serial cost model" if fit is not None else "pure N² (no model fit)"
        found.append(table)
    return found


def gustafson_fit(table: pd.DataFrame) -> Tuple[float, float]:
    """Least-squares fit of S(P) = s + p·P with s + p = 1, i.e. S - 1 = p·(P - 1). Returns (s, p)."""
    x: np.ndarray = table["P"].to_numpy(dtype=float) - 1
    y: np.ndarray = table["S"].to_numpy(dtype=float) - 1
    p: float = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 1.0
    return 1 - p, p


def plot_weak_scaling(series: List[pd.DataFrame], folder: Path) -> None:
    if not series:
        print("No weak-scaling series (N ≈ N0·√P at 2+ rank counts plus a P = 1 run); "
              "skipping weak_scaling_efficiency and gustafson_scaled_speedup")
        return

    # --- efficiency (log2 rank axis, so 1, 2, 4, ... 24 are evenly spread)
    fig, ax = new_axes()
    p_max: int = max(int(t["P"].max()) for t in series)
    all_p: List[int] = sorted({int(p) for t in series for p in t["P"]})
    ax.axhline(1, color=GUIDE, linewidth=1.2, linestyle=DASH, zorder=1)
    ax.annotate("ideal", (0.8, 1), xytext=(0, 4), textcoords="offset points",
                fontsize=8.5, color=INK_SECONDARY, ha="left", va="bottom")
    for i, table in enumerate(series):
        colour: str = MPI_COLOURS[min(i, len(MPI_COLOURS) - 1)]
        ax.plot(table["P"], table["E"], color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3,
                label=rf"$N_0$ = {thousands(table.attrs['n0'])},  $N = N_0\sqrt{{P}}$")
        if len(series) > 1:  # several series: N per point (one series: N goes under the ticks)
            for _, r in table.iterrows():
                ax.annotate(thousands(r["N"]), (r["P"], r["E"]), xytext=(0, -15 - 11 * i),
                            textcoords="offset points", ha="center", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_p)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.8, p_max * 1.25)
    if len(series) == 1:
        n_of_p: dict = dict(zip(series[0]["P"].astype(int), series[0]["N"].astype(int)))
        ax.set_xticklabels([f"{p}\n{thousands(n_of_p[p])}" for p in all_p], fontsize=8.5)
        ax.set_xlabel("MPI ranks P  (number of stars N below)")
    else:
        ax.set_xticklabels([str(p) for p in all_p])
        ax.set_xlabel("MPI ranks P")
    e_max: float = max(float(t["E"].max()) for t in series)
    ax.set_ylim(0, max(1.2, e_max * 1.1))
    ax.set_ylabel(r"Weak-scaling efficiency  $E = T(1)/T(P)$")
    ax.legend(loc="lower left")
    save(fig, folder, "weak_scaling_efficiency")

    # --- Gustafson's scaled speed-up
    fig, ax = new_axes()
    ax.plot([1, p_max], [1, p_max], color=GUIDE, linewidth=1.2, linestyle=DASH, zorder=1, label="Ideal  S = P")
    for i, table in enumerate(series):
        colour = MPI_COLOURS[min(i, len(MPI_COLOURS) - 1)]
        s_fit, p_fit = gustafson_fit(table)
        ax.plot(table["P"], table["S"], color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3,
                label=rf"$N_0$ = {thousands(table.attrs['n0'])}")
        p_line: np.ndarray = np.array([1, p_max], dtype=float)
        ax.plot(p_line, s_fit + p_fit * p_line, color=colour, linewidth=1, linestyle=(0, (1, 2)), zorder=2,
                label=rf"fit $S = s + pP$,  s = {s_fit:.3f}")
        last = table.iloc[-1]
        end_label(ax, last["P"], last["S"], f"{last['S']:.1f}")
    ax.set_xticks(all_p)
    ax.set_xlim(0, p_max * 1.12)
    ax.set_ylim(0, p_max * 1.08)
    ax.set_xlabel("MPI ranks P")
    ax.set_ylabel(r"Scaled speed-up  $S = \tau(P)/T(P)$")
    ax.legend(loc="upper left")
    save(fig, folder, "gustafson_scaled_speedup")


# ---------------------------------------------------------------- main

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Report plots from results.csv")
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--plots-dir", type=Path, default=Path("Plots"))
    parser.add_argument("--completed-only", action="store_true",
                        help="Ignore runs that stopped early (default: use them, per-step times are valid)")
    parser.add_argument("--min-steps", type=int, default=10,
                        help="Drop runs with fewer steps than this (default: 10)")
    args: argparse.Namespace = parser.parse_args()

    runs: pd.DataFrame = load_runs(args.results, args.completed_only, args.min_steps)
    pairs: pd.DataFrame = matched_pairs(runs)
    fit: Tuple[float, float] | None = fit_serial_model(runs)

    if not pairs.empty:
        print("\nSpeed-ups (serial and MPI on the same node, per time step):")
        table: pd.DataFrame = pairs[["hostname", "n_stars_start", "n_ranks", "serial_step_ms", "step_ms",
                                     "speedup", "efficiency"]].sort_values(["n_stars_start", "hostname"])
        print(table.round(3).to_string(index=False))
    if fit is not None:
        a, b = fit
        print(f"\nSerial cost model T(N) = a·N + b·N²:  a = {a * 1000:.1f} µs per star (per-row overhead), "
              f"b = {b * 1e6:.1f} ns per pair interaction")
        print(f"  the overhead and the pair work are equal at N = a/b ≈ {a / b:,.0f} stars")
    weak: List[pd.DataFrame] = weak_scaling_series(runs, fit)
    for table in weak:
        s_fit, p_fit = gustafson_fit(table)
        print(f"\nWeak scaling, N0 = {table.attrs['n0']:,} (N = N0·√P; τ from {table.attrs['tau_source']}):")
        print(table[["P", "N", "T_ms", "E", "tau_ms", "S", "repeats", "hosts"]].round(3).to_string(index=False))
        print(f"  Gustafson fit S = s + p·P:  s = {s_fit:.3f}, p = {p_fit:.3f}")
        if table["hosts"].str.split(",").explode().nunique() > 1:
            print("  NOTE: this series ran on more than one node; node speed differences (up to ~16 % seen) "
                  "end up in E. Run the whole series in one job for a clean result.")
    print()

    plot_step_time_vs_n(runs, fit, args.plots_dir)
    plot_speedup_vs_n(pairs, runs, args.plots_dir)
    plot_interactions_per_core(runs, args.plots_dir)
    plot_overhead_share(runs, args.plots_dir)
    plot_serial_vs_mpi_per_node(pairs, args.plots_dir)
    plot_speedup_vs_ranks(pairs, args.plots_dir)
    plot_weak_scaling(weak, args.plots_dir)


if __name__ == "__main__":
    main()