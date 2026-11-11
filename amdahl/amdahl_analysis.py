"""Amdahl analysis of the strong-scaling sweeps written by amdahl_batch.sh.

Reads the shared results.csv (only rows whose run_name starts with "amdahl_") and
amdahl/launch_times.csv, and writes to amdahl/out/:

    amdahl_summary.csv                  one row per version, N, time measure and P
    lost_time_summary.csv               the bars of the lost-time plots: one row per sweep and P
    amdahl_table_<label>_N<n>.tex       LaTeX table for the report (booktabs)
    speedup_N<n>.png/.pdf               speed-up per version, with the Amdahl fit
    efficiency_N<n>.png/.pdf            parallel efficiency per version
    karp_flatt_N<n>.png/.pdf            experimentally determined serial fraction per P
    measures_<label>_N<n>.png/.pdf      speed-up of the step loop, the program and the whole launch
    lost_time_<label>_N<n>.png/.pdf     where the efficiency goes, per P
    by_N_{speedup,efficiency,karp_flatt}_<label>.png/.pdf
                                        the step loop of all N in one plot (latest sweep per N)
    by_N_<label>.csv / .tex             one row per N: serial step time, S and E at the largest P,
                                        Amdahl s (and its range over all runs), s times the
                                        serial step time, Karp-Flatt
    startup_<label>.png/.pdf/.csv       Python, import and MPI start-up per launch, against P

Three wall-clock time measures:
    loop     simulation_time_s: the step loop (slowest rank per step). The main measure:
             its efficiency does not depend on how many steps we run.
    program  total_time_s: program start (after the imports) to the end, on rank 0
    launch   launch_wall_s from launch_times.csv: the whole launch, including Python
             start-up, imports and MPI start-up. The start-up is a fixed cost of about
             2 s, so its share, and with it the whole-launch efficiency, depends on how
             long the run is. It is therefore reported on its own (startup_<label>).

Amdahl:       S(P) = T(1) / T(P) = 1 / (s + (1 - s) / P)
Karp-Flatt:   e(P) = (1/S - 1/P) / (1 - 1/P)    (the s that P alone would imply)
The fitted s is the least-squares solution of 1/S - 1/P = s (1 - 1/P) over all P > 1.
The same fit to the fastest and to the slowest runs gives the range of s: how much of it is noise.
If e(P) stays flat, a fixed serial part explains the curve. If e(P) grows with P,
an overhead that grows with P (communication, waiting) is at work, which Amdahl's
model does not contain.

Usage (from the repository root):
    python3 amdahl/amdahl_analysis.py [--results results.csv] [--launch amdahl/launch_times.csv]
                                      [--out amdahl/out] [--label v1-numpy ...] [--job 12345 ...]
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NAME_RE = r"^amdahl_(?P<label>[^_]+)_j(?P<job>[^_]+)_r(?P<rep>\d+)$"
MEASURES = {"loop": "simulation_time_s", "program": "total_time_s", "launch": "launch_wall_s"}
MEASURE_TEXT = {"loop": "step loop", "program": "program (after imports)", "launch": "whole launch"}
MEASURE_SHORT = {"loop": "step loop", "program": "program", "launch": "whole launch"}   # for axis labels
MAIN = "loop"     # the measure of the tables, the per-N plots and the lost-time plot

# Colours: one per version / time measure, always in this order (checked for colour-blind
# separation on white). Serial is never a series here: it is the baseline of every speed-up.
SERIES = ["#2a78d6", "#1baf7a", "#6250d6", "#e87ba4", "#eda100"]
LOSS_PARTS = [("compute", "extra compute", "#2a78d6"),
              ("communication", "communication", "#eb6834"),
              ("waiting", "waiting and other loop work", "#6250d6"),
              ("outside", "outside the step loop (load, distribute, output)", "#1baf7a"),
              ("startup", "Python and MPI start-up", "#eda100")]
# which of these parts a measure contains
PARTS_OF = {"loop": ["compute", "communication", "waiting"],
            "program": ["compute", "communication", "waiting", "outside"],
            "launch": ["compute", "communication", "waiting", "outside", "startup"]}
INK, MUTED, GRID = "#121823", "#6f798b", "#dce1e8"
GROUP = ["label", "job", "data_file", "n_stars_start", "steps_requested"]


# ------------------------------------------------------------------ loading
def load(results: Path, launch: Path, labels: List[str], jobs: List[str]) -> pd.DataFrame:
    df = pd.read_csv(results)
    names = df["run_name"].astype(str).str.extract(NAME_RE)
    df = pd.concat([df, names], axis=1).dropna(subset=["label"])
    if labels:
        df = df[df["label"].isin(labels)]
    if jobs:
        df = df[df["job"].isin(jobs)]
    if df.empty:
        raise SystemExit(f"No matching amdahl_ rows in {results}. Run amdahl/amdahl_batch.sh first.")
    df["rep"] = df["rep"].astype(int)
    done = df["completed"].astype(str).str.lower().eq("true")
    if (~done).any():
        print(f"Ignoring {(~done).sum()} run(s) that stopped at the time limit (completed = False).")
    df = df[done].copy()
    df["launch_wall_s"] = np.nan
    if launch.exists():
        la = pd.read_csv(launch)
        la = la.loc[la["exit_code"] == 0, ["run_name", "implementation", "n_ranks", "launch_wall_s"]]
        df = df.drop(columns="launch_wall_s").merge(la, on=["run_name", "implementation", "n_ranks"], how="left")
    else:
        print(f"{launch} not found: only the 'loop' and 'program' measures are available.")
    return df


def parts(row: pd.Series) -> Dict[str, float]:
    """One run's launch time split into the parts of the lost-time plot (seconds, average rank)."""
    def section(name: str) -> float:
        v = row.get(f"{name}_time_mean_s")          # MPI rows: mean over ranks
        if pd.isna(v):
            v = row.get(f"{name}_time_s", 0.0)      # serial rows
        return 0.0 if pd.isna(v) else float(v)

    compute = section("forces") + section("integrate") + section("merge")
    comm = section("communication")
    loop = float(row["simulation_time_s"])
    return {"compute": compute, "communication": comm, "waiting": max(loop - compute - comm, 0.0),
            "outside": float(row["total_time_s"]) - loop,
            "startup": float(row["launch_wall_s"]) - float(row["total_time_s"])}


# ------------------------------------------------------------------ analysis
def fit_s(p: np.ndarray, speedup: np.ndarray) -> float:
    """Least-squares Amdahl s from the speed-ups at P > 1 (not clipped: below 0 means faster than ideal)."""
    x, y = 1 - 1 / p, 1 / speedup - 1 / p
    fit = p > 1
    return float(np.sum(x[fit] * y[fit]) / np.sum(x[fit] ** 2)) if fit.any() else np.nan


def analyse(g: pd.DataFrame) -> pd.DataFrame | None:
    """Speed-up, efficiency, Karp-Flatt and the Amdahl fit of one sweep, for every time measure."""
    serial, mpi = g[g["implementation"] == "serial"], g[g["implementation"] == "mpi"]
    if mpi.empty:
        return None
    rows = []
    for measure, col in MEASURES.items():
        if g[col].isna().all():
            continue
        base = serial if serial[col].notna().any() else mpi[mpi["n_ranks"] == 1]
        if base[col].notna().sum() == 0:
            continue
        t1 = float(base[col].median())
        per_p = mpi.groupby("n_ranks")[col].agg(["median", "min", "max", "count"]).dropna()
        p = per_p.index.to_numpy(dtype=float)
        s_of_p = t1 / per_p["median"].to_numpy()
        x, y = 1 - 1 / p, 1 / s_of_p - 1 / p
        s_raw = fit_s(p, s_of_p)
        # Amdahl's s lies between 0 and 1. Below 0 the runs scale ideally within the noise (s = 0);
        # above 1 the speed-up falls with P: an overhead, not a serial part, so there is no fit
        s_fit = max(s_raw, 0.0) if s_raw <= 1 else np.nan
        # range of s: s grows with T(P)/T(1), so the fastest runs against the slowest baseline give the
        # lowest s and the slowest runs against the fastest baseline the highest. Always brackets s_raw.
        if per_p["count"].max() > 1 or base[col].count() > 1:
            s_lo = fit_s(p, float(base[col].max()) / per_p["min"].to_numpy())
            s_hi = fit_s(p, float(base[col].min()) / per_p["max"].to_numpy())
        else:
            s_lo = s_hi = np.nan
        for k, P in enumerate(p):
            rows.append({"measure": measure, "baseline": "serial" if base is serial else "mpi P=1",
                         "t1_s": t1, "n_ranks": int(P), "t_median_s": per_p["median"].iloc[k],
                         "t_min_s": per_p["min"].iloc[k], "t_max_s": per_p["max"].iloc[k],
                         "repeats": int(per_p["count"].iloc[k]), "speedup": s_of_p[k],
                         "speedup_lo": t1 / per_p["max"].iloc[k], "speedup_hi": t1 / per_p["min"].iloc[k],
                         "efficiency": s_of_p[k] / P, "karp_flatt": y[k] / x[k] if P > 1 else np.nan,
                         "s_fit": s_fit, "s_raw": s_raw, "s_lo": s_lo, "s_hi": s_hi,
                         "max_speedup_fit": 1 / s_fit if s_fit > 0 else (np.inf if s_fit == 0 else np.nan)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plotting helpers
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.frameon": False,
                     "figure.figsize": (6.2, 4.0), "lines.linewidth": 1.8})


def style(ax, xlabel: str, ylabel: str, p_ticks) -> None:
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xticks(sorted(p_ticks))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def end_labels(ax, ends: List[tuple]) -> None:
    """Direct labels at the right end of each line, pushed apart so they don't overlap."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = 0.05 * (hi - lo)
    placed: List[float] = []
    for x, y, text, colour in sorted(ends, key=lambda e: e[1]):
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pct(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    v = 100 * v
    if v == 0:
        return "0 %"
    return f"{v:.0f} %" if abs(v) >= 10 else f"{v:.1f} %" if abs(v) >= 1 else f"{v:.2f} %" if abs(v) >= 0.01 else f"{v:.4f} %"


def fit_text(name: str, s: float) -> str:
    return f"{name}: s = {pct(s)}" if np.isfinite(s) else f"{name}: no Amdahl fit"


# ------------------------------------------------------------------ plots per N (all versions together)
def plot_versions(res: pd.DataFrame, n: int, out: Path) -> None:
    sub = res[res["n_stars_start"] == n]
    keys = list(sub[["label", "job"]].drop_duplicates().itertuples(index=False))
    dup = sub.groupby("label")["job"].nunique()
    series = []
    for i, (label, job) in enumerate(keys):
        d = sub[(sub["label"] == label) & (sub["job"] == job) & (sub["measure"] == MAIN)].sort_values("n_ranks")
        name = label if dup[label] == 1 else f"{label} (job {job})"
        series.append((name, d, SERIES[i % len(SERIES)]))
    p_all = sorted(sub["n_ranks"].unique())
    p_line = np.linspace(1, max(p_all), 200)
    measure = MEASURE_SHORT[MAIN]

    # speed-up
    fig, ax = plt.subplots()
    ax.plot(p_line, p_line, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal")
    ends = []
    for name, d, c in series:
        s = d["s_fit"].iloc[0]
        ax.errorbar(d["n_ranks"], d["speedup"], yerr=[d["speedup"] - d["speedup_lo"], d["speedup_hi"] - d["speedup"]],
                    color=c, marker="o", ms=5, capsize=2, label=name)
        if np.isfinite(s):
            ax.plot(p_line, 1 / (s + (1 - s) / p_line), color=c, lw=1, ls=(0, (2, 2)))
        ends.append((d["n_ranks"].iloc[-1], d["speedup"].iloc[-1], fit_text(name, s), c))
    ax.plot([], [], color=MUTED, lw=1, ls=(0, (2, 2)), label="Amdahl fit")
    style(ax, "MPI ranks P", f"Speed-up T(serial) / T(P), {measure}", p_all)
    ax.set_ylim(0, max(p_all) * 1.05)
    ax.legend(loc="upper left")
    end_labels(ax, ends)
    save(fig, out, f"speedup_N{n}")

    # efficiency
    fig, ax = plt.subplots()
    ax.axhline(100, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal")
    for name, d, c in series:
        s = d["s_fit"].iloc[0]
        ax.plot(d["n_ranks"], 100 * d["efficiency"], color=c, marker="o", ms=5, label=name)
        if np.isfinite(s):
            ax.plot(p_line, 100 / (s * p_line + 1 - s), color=c, lw=1, ls=(0, (2, 2)))
    ax.plot([], [], color=MUTED, lw=1, ls=(0, (2, 2)), label="Amdahl fit")
    style(ax, "MPI ranks P", f"Efficiency S / P (%), {measure}", p_all)
    ax.set_ylim(0, max(110, 100 * max(d["efficiency"].max() for _, d, _ in series) + 5))
    ax.legend(loc="lower left")
    save(fig, out, f"efficiency_N{n}")

    # Karp-Flatt
    fig, ax = plt.subplots()
    for name, d, c in series:
        d = d[d["n_ranks"] > 1]
        if d.empty:
            continue
        ax.plot(d["n_ranks"], 100 * d["karp_flatt"], color=c, marker="o", ms=5, label=name)
        if np.isfinite(d["s_fit"].iloc[0]):
            ax.axhline(100 * d["s_fit"].iloc[0], color=c, lw=1, ls=(0, (2, 2)))
    ax.plot([], [], color=MUTED, lw=1, ls=(0, (2, 2)), label="fitted Amdahl s")
    style(ax, "MPI ranks P", f"Karp-Flatt e(P) (%), {measure}", [q for q in p_all if q > 1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    save(fig, out, f"karp_flatt_N{n}")


# ------------------------------------------------------------------ plots per sweep
def plot_measures(res: pd.DataFrame, tag: str, out: Path) -> None:
    """Speed-up of the step loop, the program and the whole launch: what the parts outside the loop cost."""
    fig, ax = plt.subplots()
    p_all = sorted(res["n_ranks"].unique())
    ax.plot(p_all, p_all, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal")
    ends = []
    for i, measure in enumerate(m for m in MEASURES if (res["measure"] == m).any()):
        d = res[res["measure"] == measure].sort_values("n_ranks")
        ax.plot(d["n_ranks"], d["speedup"], color=SERIES[i], marker="o", ms=5, label=MEASURE_TEXT[measure])
        ends.append((d["n_ranks"].iloc[-1], d["speedup"].iloc[-1], fit_text(MEASURE_TEXT[measure], d["s_fit"].iloc[0]),
                     SERIES[i]))
    style(ax, "MPI ranks P", "Speed-up S = T(serial) / T(P)", p_all)
    ax.set_ylim(0, max(p_all) * 1.05)
    ax.legend(loc="upper left")
    end_labels(ax, ends)
    save(fig, out, f"measures_{tag}")


def plot_lost_time(g: pd.DataFrame, res: pd.DataFrame, tag: str, out: Path) -> pd.DataFrame:
    """Lost core time P·T(P) − T(serial), split by cause, in % of the serial run (stacks to 1/E − 1).

    Returns the bars as a table: one row per P, the parts in % of the serial run, and the
    communication time itself (seconds, mean over the ranks, median over the repeats)."""
    measure = MAIN
    serial, mpi = g[g["implementation"] == "serial"], g[g["implementation"] == "mpi"]
    base = serial if not serial.empty else mpi[mpi["n_ranks"] == 1]
    use = [p for p in LOSS_PARTS if p[0] in PARTS_OF[measure]]
    split = pd.DataFrame([parts(r) for _, r in g.iterrows()], index=g.index)
    base_parts = split.loc[base.index].median()
    t1 = float(res.loc[res["measure"] == measure, "t1_s"].iloc[0])
    ps = sorted(mpi["n_ranks"].unique())
    lost = pd.DataFrame({P: (P * split.loc[mpi.index[(mpi["n_ranks"] == P).to_numpy()]].median() - base_parts) / t1 * 100
                         for P in ps}).T
    total = res[res["measure"] == measure].set_index("n_ranks")["efficiency"]
    fig, ax = plt.subplots()
    xs = np.arange(len(ps))
    pos, neg = np.zeros(len(ps)), np.zeros(len(ps))
    for key, text, colour in use:
        v = lost[key].to_numpy()
        bottom = np.where(v >= 0, pos, neg)
        ax.bar(xs, v, 0.62, bottom=bottom, color=colour, edgecolor="white", linewidth=1, label=text)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.plot(xs, [100 * (1 / total[P] - 1) for P in ps], ls="none", marker="_", ms=16, mew=2, color=INK,
            label="total = 1/E − 1")
    ax.axhline(0, color=MUTED, lw=1)
    style(ax, "MPI ranks P", f"Lost core time (% of serial), {MEASURE_SHORT[measure]}", [])
    ax.set_xticks(xs, [str(P) for P in ps])
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, f"lost_time_{tag}")
    table = lost[[key for key, _, _ in use]].rename(columns=lambda k: f"lost_{k}_pct")
    table["lost_total_pct"] = [100 * (1 / total[P] - 1) for P in ps]
    table["communication_s"] = [float(split.loc[mpi.index[(mpi["n_ranks"] == P).to_numpy()], "communication"].median())
                                for P in ps]
    return table.rename_axis("n_ranks").reset_index()


# ------------------------------------------------------------------ all N of one version together
def latest_sweeps(res: pd.DataFrame, label: str, measure: str = "loop") -> List[tuple]:
    """(N, rows of that sweep) for every N of one version; the latest job if an N was run more than once."""
    lr = res[(res["label"] == label) & (res["measure"] == measure)]
    sweeps = []
    for n, g in lr.groupby("n_stars_start"):
        job = max(g["job"].unique(), key=lambda j: (len(str(j)), str(j)))
        sweeps.append((int(n), g[g["job"] == job].sort_values("n_ranks")))
    return sweeps


def plot_by_n(res: pd.DataFrame, label: str, out: Path, measure: str = "loop") -> None:
    """Speed-up, efficiency and Karp-Flatt of the step loop with one line per N: where strong scaling breaks down."""
    sweeps = latest_sweeps(res, label, measure)
    if len(sweeps) < 2:
        return
    p_all = sorted({int(p) for _, d in sweeps for p in d["n_ranks"]})
    what = MEASURE_SHORT[measure]
    ylabels = {"speedup": f"Speed-up T(serial) / T(P), {what}", "efficiency": f"Efficiency S / P (%), {what}",
               "karp_flatt": f"Karp-Flatt e(P) (%), {what}"}
    for kind, ylabel in ylabels.items():
        fig, ax = plt.subplots()
        if kind == "speedup":
            ax.plot(p_all, p_all, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal")
        elif kind == "efficiency":
            ax.axhline(100, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal")
        else:
            ax.axhline(0, color=MUTED, lw=1)
        top = 0.0
        for i, (n, d) in enumerate(sweeps):
            if kind == "karp_flatt":
                d = d[d["n_ranks"] > 1]
            y = d[kind] * (1 if kind == "speedup" else 100)
            top = max(top, float(y.max()))
            ax.plot(d["n_ranks"], y, color=SERIES[i % len(SERIES)], marker="o", ms=5, label=f"N = {n:,}")
        style(ax, "MPI ranks P", ylabel, [p for p in p_all if kind != "karp_flatt" or p > 1])
        if kind == "speedup":
            ax.set_ylim(0, max(p_all) * 1.05)
            ax.legend(loc="upper left")
        elif kind == "efficiency":
            ax.set_ylim(0, max(110, top + 5))
            ax.legend(loc="lower left")
        else:
            ax.legend(loc="best")
        save(fig, out, f"by_N_{kind}_{label}")


def by_n_table(res: pd.DataFrame, label: str, out: Path, measure: str = "loop") -> pd.DataFrame | None:
    """One row per N: how much of a step the Amdahl fit treats as serial, and whether Karp-Flatt rises."""
    sweeps = latest_sweeps(res, label, measure)
    if not sweeps:
        return None
    rows = []
    for n, d in sweeps:
        steps = int(d["steps_requested"].iloc[0])
        t_step = 1000 * float(d["t1_s"].iloc[0]) / steps
        kf = d[d["n_ranks"] > 1].set_index("n_ranks")["karp_flatt"]
        p_kf = next((p for p in kf.index if p >= 4), kf.index[0]) if len(kf) else np.nan
        s = float(d["s_fit"].iloc[0])
        last = d.iloc[-1]
        rows.append({"label": label, "n_stars": n, "steps": steps, "job": d["job"].iloc[0],
                     "serial_ms_per_step": t_step, "p_max": int(last["n_ranks"]),
                     "speedup_pmax": float(last["speedup"]), "efficiency_pmax": float(last["efficiency"]),
                     "s_fit": s, "s_lo": float(d["s_lo"].iloc[0]), "s_hi": float(d["s_hi"].iloc[0]),
                     "serial_ms_per_step_fit": s * t_step if np.isfinite(s) else np.nan,
                     "p_kf_first": p_kf, "karp_flatt_first": kf.get(p_kf, np.nan) if len(kf) else np.nan,
                     "karp_flatt_pmax": kf.iloc[-1] if len(kf) else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(out / f"by_N_{label}.csv", index=False)

    pk = int(t["p_kf_first"].dropna().iloc[0]) if t["p_kf_first"].notna().any() else 4
    pm = int(t["p_max"].max())
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{Strong scaling of the step loop of \texttt{{{label}}} for different $N$ (one fully "
             r"reserved node per $N$, medians over repeats). $T_\mathrm{step}$: serial time per step. "
             r"$s$: Amdahl fit, in brackets the same fit to the fastest and to the slowest runs; "
             r"$s\,T_\mathrm{step}$ is the time per step that the fit treats as serial. "
             r"$e$: Karp--Flatt serial fraction; flat means a fixed serial part (Amdahl), rising means an "
             r"overhead that grows with $P$.}",
             rf"  \label{{tab:amdahl_by_n_{label}}}", r"  \begin{tabular}{rrrrrrrrr}", r"    \toprule",
             rf"    $N$ & steps & $T_\mathrm{{step}}$ [ms] & $S({pm})$ & $E({pm})$ & $s$ & (runs) & "
             rf"$s\,T_\mathrm{{step}}$ [ms] & $e({pk}) \to e({pm})$ \\", r"    \midrule"]
    def num(v: float) -> str:      # 3 significant digits without e-notation, real minus sign
        if not np.isfinite(v):
            return "---"
        txt = f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"
        return f"${txt}$"

    for _, r in t.iterrows():
        rng = (rf"({num(100 * r['s_lo'])} to {num(100 * r['s_hi'])}\,\%)" if np.isfinite(r["s_lo"])
               else "---")
        lines.append(rf"    {r['n_stars']:,} & {r['steps']:,} & {num(r['serial_ms_per_step'])} & "
                     rf"{r['speedup_pmax']:.1f} & {100 * r['efficiency_pmax']:.0f}\,\% & "
                     rf"{num(100 * r['s_fit'])}\,\% & {rng} & {num(r['serial_ms_per_step_fit'])} & "
                     rf"{num(100 * r['karp_flatt_first'])}\,\% $\to$ {num(100 * r['karp_flatt_pmax'])}\,\% \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / f"by_N_{label}.tex").write_text("\n".join(lines) + "\n")
    return t


def print_by_n(t: pd.DataFrame) -> None:
    label = t["label"].iloc[0]
    print(f"\n=== {label}: step loop for every N (latest sweep per N)")
    print(f"  {'N':>6} {'steps':>7} {'serial/step':>12} {'S(Pmax)':>8} {'E(Pmax)':>8} {'Amdahl s':>9} "
          f"{'s from all runs':>20} {'s*T_step':>10}   Karp-Flatt")
    for _, r in t.iterrows():
        st = f"{r['serial_ms_per_step_fit']:.3g} ms" if np.isfinite(r["serial_ms_per_step_fit"]) else "n/a"
        rng = f"{pct(r['s_lo'])} .. {pct(r['s_hi'])}" if np.isfinite(r["s_lo"]) else "n/a"
        print(f"  {r['n_stars']:>6,} {r['steps']:>7,} {r['serial_ms_per_step']:>9.4g} ms {r['speedup_pmax']:>8.1f} "
              f"{100 * r['efficiency_pmax']:>7.0f}% {pct(r['s_fit']):>9} {rng:>20} {st:>10}   "
              f"{pct(r['karp_flatt_first'])} (P = {r['p_kf_first']:.0f}) -> {pct(r['karp_flatt_pmax'])} (P = {r['p_max']})")
    print("  s*T_step: the time per step that Amdahl's fit treats as serial. Karp-Flatt rising with P means an\n"
          "  overhead that grows with P (communication latency, uneven split of the stars), which Amdahl does not model.\n"
          "  s from all runs: the fit to the fastest and to the slowest runs. Where it reaches below 0, s is not distinguishable from 0.")


# ------------------------------------------------------------------ start-up
def startup(df: pd.DataFrame, label: str, out: Path) -> pd.DataFrame | None:
    """Python, import and MPI start-up of every launch: launch wall time minus the program's own total time.

    It does not depend on N, so all sweeps of one version go into one plot, with a straight line
    fitted to the mpirun launches: start-up(P) = a + b P.
    """
    d = df[(df["label"] == label) & df["launch_wall_s"].notna()].copy()
    if d.empty:
        return None
    d["startup_s"] = d["launch_wall_s"] - d["total_time_s"]
    t = (d.groupby(["n_stars_start", "implementation", "n_ranks"])["startup_s"]
         .agg(["median", "min", "max", "count"]).reset_index())
    t.insert(0, "label", label)
    mpi, ser = t[t["implementation"] == "mpi"], t[t["implementation"] == "serial"]
    b, a = np.polyfit(mpi["n_ranks"], mpi["median"], 1) if mpi["n_ranks"].nunique() > 1 else (np.nan, np.nan)
    t.attrs.update(a=a, b=b)
    t.to_csv(out / f"startup_{label}.csv", index=False)

    fig, ax = plt.subplots()
    p_all = sorted(int(p) for p in mpi["n_ranks"].unique())
    for i, (n, g) in enumerate(mpi.groupby("n_stars_start")):
        ax.plot(g["n_ranks"], g["median"], color=SERIES[i % len(SERIES)], marker="o", ms=5, label=f"N = {int(n):,}")
    if np.isfinite(a):
        ax.plot(p_all, a + b * np.array(p_all), color=MUTED, lw=1, ls=(0, (2, 2)),
                label=f"fit: {a:.2f} s + {1000 * b:.0f} ms per rank")
    if len(ser):
        ax.plot(np.ones(len(ser)), ser["median"], ls="none", marker="D", ms=5, color=INK,
                label="serial program (no mpirun)")
    style(ax, "MPI ranks P", "Start-up per launch (s)", p_all)
    ax.set_ylim(0, 1.2 * float(t["median"].max()))
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, out, f"startup_{label}")
    return t


def print_startup(t: pd.DataFrame) -> None:
    label = t["label"].iloc[0]
    print(f"\n=== {label}: start-up per launch (Python, imports, MPI) = launch wall time - the program's own time")
    p_max = int(t.loc[t["implementation"] == "mpi", "n_ranks"].max())
    print(f"  {'N':>6} {'serial':>8} {'P = 1':>8} {f'P = {p_max}':>8}")
    for n, g in t.groupby("n_stars_start"):
        g = g.set_index(["implementation", "n_ranks"])["median"]
        print(f"  {int(n):>6,} {g.get(('serial', 1), np.nan):7.2f}s {g.get(('mpi', 1), np.nan):7.2f}s "
              f"{g.get(('mpi', p_max), np.nan):7.2f}s")
    if np.isfinite(t.attrs.get("a", np.nan)):
        print(f"  mpirun launches: {t.attrs['a']:.2f} s + {1000 * t.attrs['b']:.0f} ms per rank, the same for every N.\n"
              f"  A fixed cost per run: it matters in short runs like these sweeps, not in long production runs.")


# ------------------------------------------------------------------ report outputs
def latex_table(res: pd.DataFrame, info: pd.Series, out: Path, tag: str) -> None:
    measure = MAIN
    d = res[res["measure"] == measure].sort_values("n_ranks")
    s = d["s_fit"].iloc[0]
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{Strong scaling of \texttt{{{info['label']}}} on {int(info['n_stars_start']):,} stars, "
             rf"{int(info['steps_requested'])} steps, one node ({info['hostname']}, job {info['job']}). "
             rf"Times are medians over repeats of the {MEASURE_TEXT[measure]}. "
             rf"$S = T_\mathrm{{serial}}/T(P)$, $E = S/P$, $e$ is the Karp--Flatt serial fraction. "
             + (rf"Amdahl fit: $s = {100 * s:.3g}\,\%$, $S_\mathrm{{max}} = 1/s = {1 / s:.3g}$.}}" if np.isfinite(s) and s > 0
                else r"Amdahl fit: no measurable serial part ($s \approx 0$).}" if s == 0
                else r"No Amdahl fit: the speed-up falls with $P$, so an overhead dominates, not a fixed serial part.}"),
             rf"  \label{{tab:amdahl_{tag}}}", r"  \begin{tabular}{rrrrr}", r"    \toprule",
             r"    $P$ & $T$ [s] & $S$ & $E$ & $e$ \\", r"    \midrule",
             rf"    serial & {d['t1_s'].iloc[0]:.2f} & --- & --- & --- \\"]
    for _, r in d.iterrows():
        e = "---" if r["n_ranks"] == 1 else f"{100 * r['karp_flatt']:.3g}\\,\\%"
        lines.append(rf"    {int(r['n_ranks'])} & {r['t_median_s']:.2f} & {r['speedup']:.2f} & "
                     rf"{100 * r['efficiency']:.1f}\,\% & {e} \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / f"amdahl_table_{tag}.tex").write_text("\n".join(lines) + "\n")


def print_report(res: pd.DataFrame, info: pd.Series) -> None:
    print(f"\n=== {info['label']}, N = {int(info['n_stars_start']):,}, {int(info['steps_requested'])} steps, "
          f"job {info['job']} on {info['hostname']}")
    for measure in MEASURES:
        d = res[res["measure"] == measure].sort_values("n_ranks")
        if d.empty:
            continue
        s = d["s_fit"].iloc[0]
        fit = (f"Amdahl s = {pct(s):>8}   S_max = 1/s = {d['max_speedup_fit'].iloc[0]:.3g}" if np.isfinite(s)
               else f"no Amdahl fit (least squares gives s = {pct(d['s_raw'].iloc[0])}: the speed-up falls with P)")
        if np.isfinite(d["s_lo"].iloc[0]):
            fit += f"   (runs: {pct(d['s_lo'].iloc[0])} .. {pct(d['s_hi'].iloc[0])})"
        print(f"  {MEASURE_TEXT[measure]:<24} baseline {d['baseline'].iloc[0]} {d['t1_s'].iloc[0]:8.2f} s   {fit}")
    d = res[res["measure"] == MAIN].set_index("n_ranks").sort_index()
    print(f"  {'P':>4} {'T [s]':>9} {'S':>7} {'E':>7} {'e(P)':>9}  repeats   ({MEASURE_TEXT[MAIN]})")
    for P, r in d.iterrows():
        e = "" if P == 1 else pct(r["karp_flatt"])
        print(f"  {P:>4} {r['t_median_s']:9.2f} {r['speedup']:7.2f} {100 * r['efficiency']:6.1f}% {e:>9}  {r['repeats']:>7}")
    if 1 in d.index and d["baseline"].iloc[0] == "serial" and abs(d.loc[1, "speedup"] - 1) > 0.02:
        print(f"  the MPI program at P = 1 is {abs(d.loc[1, 'speedup'] - 1):.0%} "
              f"{'faster' if d.loc[1, 'speedup'] > 1 else 'slower'} than the serial program (same force kernel); "
              f"S and E are relative to the serial program")
    gains = [f"{P}→{2 * P}: ×{d.loc[2 * P, 'speedup'] / d.loc[P, 'speedup']:.2f}" for P in d.index if 2 * P in d.index]
    if gains:
        print("  doubling the ranks gives  " + "   ".join(gains))
    kf = d.loc[d.index > 1, "karp_flatt"]
    if len(kf) >= 2 and kf.iloc[0] > 0 and kf.iloc[-1] > 2 * kf.iloc[0]:
        print(f"  Karp-Flatt e(P) grows from {pct(kf.iloc[0])} (P = {kf.index[0]}) to {pct(kf.iloc[-1])} "
              f"(P = {kf.index[-1]}): an overhead that grows with P, not only a fixed serial part")
    below = d.index[d["efficiency"] < 0.7]
    print(f"  efficiency drops below 70 % at P = {below[0]}" if len(below) else "  efficiency stays above 70 % for all P")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=REPO / "results.csv")
    ap.add_argument("--launch", type=Path, default=HERE / "launch_times.csv")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--label", nargs="*", default=[], help="only these code versions")
    ap.add_argument("--job", nargs="*", default=[], help="only these LSF job IDs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load(args.results, args.launch, args.label, args.job)
    all_res, all_lost = [], []
    for key, g in df.groupby(GROUP):
        res = analyse(g)
        if res is None:
            continue
        info = pd.Series(dict(zip(GROUP, key)) | {"hostname": g["hostname"].iloc[0]})
        for col in GROUP:
            res[col] = info[col]
        res["hostname"] = info["hostname"]
        tag = f"{info['label']}_N{int(info['n_stars_start'])}"
        if df[(df["label"] == info["label"]) & (df["n_stars_start"] == info["n_stars_start"])]["job"].nunique() > 1:
            tag += f"_j{info['job']}"
        print_report(res, info)
        plot_measures(res, tag, args.out)
        lost = plot_lost_time(g, res, tag, args.out)
        for col in GROUP:
            lost.insert(GROUP.index(col), col, info[col])
        all_lost.append(lost)
        latex_table(res, info, args.out, tag)
        all_res.append(res)
    if not all_res:
        raise SystemExit("No sweep with MPI runs found.")
    res = pd.concat(all_res, ignore_index=True)
    for n in sorted(res["n_stars_start"].unique()):
        plot_versions(res, int(n), args.out)
    for label in sorted(res["label"].unique()):
        t = by_n_table(res, label, args.out)
        if t is not None and len(t) > 1:
            plot_by_n(res, label, args.out)
            print_by_n(t)
    for label in sorted(df["label"].unique()):
        t = startup(df, label, args.out)
        if t is not None:
            print_startup(t)
    res[GROUP + [c for c in res.columns if c not in GROUP]].to_csv(args.out / "amdahl_summary.csv", index=False)
    pd.concat(all_lost, ignore_index=True).to_csv(args.out / "lost_time_summary.csv", index=False)
    print(f"\nWrote plots, LaTeX tables, amdahl_summary.csv and lost_time_summary.csv to {args.out}")


if __name__ == "__main__":
    main()
