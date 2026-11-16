"""Gustafson analysis of the weak-scaling sweeps written by gustafson_batch.sh.

Reads the shared results.csv (only rows whose run_name starts with "gustafson_") and
gustafson/launch_times.csv, and writes to gustafson/out/:

    gustafson_summary.csv                one row per sweep, time measure and P
    lost_time_summary.csv                the bars of the lost-time plots: one row per sweep and P
    by_N0_<label>.csv / .tex             one row per sweep: E, S and the Gustafson s at the largest P
    gustafson_table_<label>_<rule>_N0_<n>.tex   LaTeX table of one sweep (booktabs)
    scaled_speedup_<label>_<rule>.png/.pdf    S(P) = tau(P) / T(P), one line per N0, Gustafson fit
    weak_efficiency_<label>_<rule>.png/.pdf   E(P) = T(1) / T(P), one line per N0
    gustafson_fraction_<label>_<rule>.png/.pdf  s(P) = (P - S) / (P - 1), the serial fraction each P implies
    growth_<label>.png/.pdf              tau(P) / tau(1): how much more serial work the scaled problem is
    rules_<label>_N0_<n>.png/.pdf        the two growth rules side by side (only if both ran for that N0)
    measures_<label>_<rule>_N0_<n>.png/.pdf  weak efficiency of the step loop, the program and the whole launch
    lost_time_<label>_<rule>_N0_<n>.png/.pdf where the time per step goes, per P
    startup_<label>.csv                  start-up per launch (whole launch minus the program's own time)

The quantities (per time step, so runs with different numbers of steps compare):
    N(P)     stars of the problem of P ranks; N(1) = N0 (see gustafson/plan.py for the two rules)
    T(P)     step loop of the MPI program on N(P) stars with P ranks (simulation_time_s / iterations,
             the slowest rank per step)
    tau(P)   step loop of the serial program on N(P) stars: the serial time of the scaled problem.
             T(1) = tau(1) is the serial program on N0 stars.
    E(P)     = T(1) / T(P)     weak-scaling efficiency (week 8, slide 5); ideal 1
    S(P)     = tau(P) / T(P)   Gustafson's scaled speed-up; ideal P
    g(P)     = tau(P) / tau(1) how much the serial work grew; ideal P. S = g * E
    s(P)     = (P - S) / (P - 1)  the serial fraction that S at this P implies on its own
Gustafson:   S(P) = s + (1 - s) P, with s the serial fraction of the PARALLEL run's time
             (Amdahl's s is a fraction of the serial run's time). The fitted s is the least-squares
             solution of P - S = s (P - 1) over all P > 1. The same fit to the fastest and to the
             slowest runs gives its range. s(P) flat: a fixed serial part per step, as Gustafson
             assumes. s(P) rising: a cost that grows with P (communication, waiting).

The program and whole-launch measures compare whole runs, which only works where the runs have
the same number of steps: the MPI runs at every P and the serial run on N0. They give E only.

Usage (from the repository root):
    python3 gustafson/gustafson_analysis.py [--results results.csv] [--launch gustafson/launch_times.csv]
                                            [--out gustafson/out] [--label v1-numpy ...] [--job 12345 ...]
"""
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NAME_RE = r"^gustafson_(?P<label>[^_]+)_(?P<rule>time|pairs)(?P<n0>\d+)_j(?P<job>[^_]+)_r(?P<rep>\d+)$"
GROUP = ["label", "rule", "n0", "job"]
MEASURES = {"loop": "simulation_time_s", "program": "total_time_s", "launch": "launch_wall_s"}
MEASURE_TEXT = {"loop": "step loop", "program": "program (after imports)", "launch": "whole launch"}
RULE_TEXT = {"time": "same serial time per rank", "pairs": "N0·√P, same pairs per rank"}
MAIN_RULE = "time"

# Same colours as amdahl_analysis.py (checked for colour-blind separation on white)
SERIES = ["#2a78d6", "#1baf7a", "#6250d6", "#e87ba4", "#eda100"]
LOSS_PARTS = [("compute", "compute (own rows)", "#2a78d6"),
              ("communication", "communication", "#eb6834"),
              ("waiting", "waiting and other loop work", "#6250d6")]
INK, MUTED, GRID = "#121823", "#6f798b", "#dce1e8"


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
        raise SystemExit(f"No matching gustafson_ rows in {results}. Run gustafson/gustafson_batch.sh first.")
    df["rep"] = df["rep"].astype(int)
    df["n0"] = df["n0"].astype(int)
    done = df["completed"].astype(str).str.lower().eq("true")
    if (~done).any():
        print(f"Ignoring {(~done).sum()} run(s) that stopped at the time limit (completed = False).")
    df = df[done & (df["iterations"] > 0)].copy()
    df["launch_wall_s"] = np.nan
    if launch.exists():
        la = pd.read_csv(launch)
        la = la.loc[la["exit_code"] == 0, ["run_name", "implementation", "n_ranks", "data_file", "launch_wall_s"]]
        df = df.drop(columns="launch_wall_s").merge(la, on=["run_name", "implementation", "n_ranks", "data_file"],
                                                    how="left")
    else:
        print(f"{launch} not found: the whole-launch measure and the start-up are not available.")
    # per-step times (ms) and the parts of a step (ms, mean over the ranks)
    it = df["iterations"].astype(float)
    df["loop_ms"] = 1000 * df["simulation_time_s"] / it

    def section(name: str) -> pd.Series:
        mean = df.get(f"{name}_time_mean_s", pd.Series(np.nan, index=df.index))
        return mean.fillna(df[f"{name}_time_s"]).fillna(0.0)
    df["compute_ms"] = 1000 * (section("forces") + section("integrate") + section("merge")) / it
    df["comm_ms"] = 1000 * section("communication") / it
    df["waiting_ms"] = (df["loop_ms"] - df["compute_ms"] - df["comm_ms"]).clip(lower=0)
    return df


def scaled_p(g: pd.DataFrame) -> pd.Series:
    """P of the problem each run belongs to: MPI runs have it as n_ranks, a serial run gets the P
    of the MPI runs on the same N (N0 is P = 1)."""
    mpi = g[g["implementation"] == "mpi"]
    p_of_n = dict(zip(mpi["n_stars_start"], mpi["n_ranks"]))
    p_of_n[int(g["n0"].iloc[0])] = 1
    p = g["n_stars_start"].map(p_of_n)
    p[g["implementation"] == "mpi"] = g["n_ranks"]
    return p


# ------------------------------------------------------------------ analysis
def fit_s(p: np.ndarray, speedup: np.ndarray) -> float:
    """Least-squares Gustafson s from P - S = s (P - 1) over P > 1 (not clipped)."""
    fit = p > 1
    x, y = p[fit] - 1, p[fit] - speedup[fit]
    return float(np.sum(x * y) / np.sum(x * x)) if fit.any() else np.nan


def analyse(g: pd.DataFrame) -> pd.DataFrame | None:
    """E, S, g and s of one sweep: the step loop per step, the program and the launch per run."""
    g = g.assign(P=scaled_p(g)).dropna(subset=["P"])
    serial, mpi = g[g["implementation"] == "serial"], g[g["implementation"] == "mpi"]
    base = serial[serial["P"] == 1]
    if mpi.empty or base.empty:
        return None
    n0 = int(g["n0"].iloc[0])
    t = mpi.groupby("P")["loop_ms"].agg(["median", "min", "max", "count"])
    tau = serial.groupby("P")["loop_ms"].agg(["median", "min", "max", "count"])
    n_of = mpi.groupby("P")["n_stars_start"].first()
    steps = mpi.groupby("P")["steps_requested"].first()
    s_steps = serial.groupby("P")["steps_requested"].first()
    ps = np.array(sorted(t.index), dtype=float)
    t1 = float(tau.loc[1, "median"])
    tau_m = tau["median"].reindex(ps).to_numpy()
    T = t["median"].reindex(ps).to_numpy()
    S = tau_m / T
    S_lo = tau["min"].reindex(ps).to_numpy() / t["max"].reindex(ps).to_numpy()
    S_hi = tau["max"].reindex(ps).to_numpy() / t["min"].reindex(ps).to_numpy()
    ok = np.isfinite(S)
    s_fit = fit_s(ps[ok], S[ok])
    s_lo = fit_s(ps[ok], S_hi[ok]) if (t["count"].max() > 1 or tau["count"].max() > 1) else np.nan
    s_hi = fit_s(ps[ok], S_lo[ok]) if np.isfinite(s_lo) else np.nan
    rows = []
    for k, P in enumerate(ps):
        n = int(n_of.loc[P])
        rows.append({
            "measure": "loop", "n_ranks": int(P), "n_stars": n, "steps": int(steps.loc[P]),
            "serial_steps": int(s_steps.loc[P]) if P in s_steps.index else np.nan,
            "repeats": int(t.loc[P, "count"]), "serial_repeats": int(tau.loc[P, "count"]) if P in tau.index else 0,
            "t1_ms": t1, "t_ms": T[k], "t_min_ms": t.loc[P, "min"], "t_max_ms": t.loc[P, "max"],
            "tau_ms": tau_m[k], "tau_min_ms": tau["min"].get(P, np.nan), "tau_max_ms": tau["max"].get(P, np.nan),
            "efficiency": t1 / T[k], "efficiency_lo": t1 / t.loc[P, "max"], "efficiency_hi": t1 / t.loc[P, "min"],
            "speedup": S[k], "speedup_lo": S_lo[k], "speedup_hi": S_hi[k],
            "scaled_efficiency": S[k] / P, "growth": tau_m[k] / t1,
            "pairs_per_rank": n * (n - 1) / P / (n0 * (n0 - 1)),
            "s_of_p": (P - S[k]) / (P - 1) if P > 1 else np.nan,
            "s_fit": s_fit, "s_lo": s_lo, "s_hi": s_hi})
    # whole runs: only the MPI runs (same steps at every P) against the serial run on N0
    for measure in ("program", "launch"):
        col = MEASURES[measure]
        if g[col].isna().all() or base[col].isna().all():
            continue
        b = float(base[col].median())
        w = mpi.groupby("P")[col].agg(["median", "min", "max", "count"]).dropna()
        for P, r in w.iterrows():
            rows.append({"measure": measure, "n_ranks": int(P), "n_stars": int(n_of.loc[P]),
                         "steps": int(steps.loc[P]), "repeats": int(r["count"]), "t1_ms": 1000 * b,
                         "t_ms": 1000 * r["median"], "t_min_ms": 1000 * r["min"], "t_max_ms": 1000 * r["max"],
                         "efficiency": b / r["median"], "efficiency_lo": b / r["max"], "efficiency_hi": b / r["min"]})
    return pd.DataFrame(rows)


def lost_time(g: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Extra time per step against the base problem, T(P) - T(1), in % of T(1) (= 1/E - 1),
    split into compute of the own rows, communication and waiting (means over the ranks)."""
    g = g.assign(P=scaled_p(g))
    base = g[(g["implementation"] == "serial") & (g["P"] == 1)]
    mpi = g[g["implementation"] == "mpi"]
    loop = res[res["measure"] == "loop"].set_index("n_ranks")
    t1 = float(loop["t1_ms"].iloc[0])
    b = base[["compute_ms", "comm_ms", "waiting_ms"]].median()
    rows = []
    for P, m in mpi.groupby("P"):
        med = m[["compute_ms", "comm_ms", "waiting_ms"]].median()
        rows.append({"n_ranks": int(P), "n_stars": int(m["n_stars_start"].iloc[0]),
                     "lost_compute_pct": 100 * (med["compute_ms"] - b["compute_ms"]) / t1,
                     "lost_communication_pct": 100 * (med["comm_ms"] - b["comm_ms"]) / t1,
                     "lost_waiting_pct": 100 * (med["waiting_ms"] - b["waiting_ms"]) / t1,
                     "lost_total_pct": 100 * (1 / loop.loc[int(P), "efficiency"] - 1),
                     "compute_ms": med["compute_ms"], "communication_ms": med["comm_ms"],
                     "waiting_ms": med["waiting_ms"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plotting helpers
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.frameon": False,
                     "figure.figsize": (6.2, 4.0), "lines.linewidth": 1.8})


def style(ax, xlabel: str, ylabel: str, p_ticks) -> None:
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if len(p_ticks):
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
    for x, y, text in sorted(ends, key=lambda e: e[1]):
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
    return f"{v:.0f} %" if abs(v) >= 10 else f"{v:.1f} %" if abs(v) >= 1 else f"{v:.2f} %"


# ------------------------------------------------------------------ plots of one rule, one line per N0
def latest(res: pd.DataFrame, label: str, rule: str) -> List[tuple]:
    """(N0, loop rows of that sweep) for every N0 of one version and rule; the latest job if run twice."""
    lr = res[(res["label"] == label) & (res["rule"] == rule) & (res["measure"] == "loop")]
    out = []
    for n0, g in lr.groupby("n0"):
        job = max(g["job"].unique(), key=lambda j: (len(str(j)), str(j)))
        out.append((int(n0), g[g["job"] == job].sort_values("n_ranks")))
    return out


def plot_rule(res: pd.DataFrame, label: str, rule: str, out: Path) -> None:
    sweeps = latest(res, label, rule)
    if not sweeps:
        return
    p_all = sorted({int(p) for _, d in sweeps for p in d["n_ranks"]})
    p_line = np.linspace(1, max(p_all), 100)
    tag = f"{label}_{rule}"

    # scaled speed-up with the Gustafson fit
    fig, ax = plt.subplots()
    ax.plot(p_all, p_all, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal S = P")
    ends = []
    for i, (n0, d) in enumerate(sweeps):
        c, s = SERIES[i % len(SERIES)], d["s_fit"].iloc[0]
        ax.errorbar(d["n_ranks"], d["speedup"], yerr=[d["speedup"] - d["speedup_lo"], d["speedup_hi"] - d["speedup"]],
                    color=c, marker="o", ms=5, capsize=2, label=f"N0 = {n0:,}")
        if np.isfinite(s):
            ax.plot(p_line, s + (1 - s) * p_line, color=c, lw=1, ls=(0, (2, 2)))
        ends.append((d["n_ranks"].iloc[-1], d["speedup"].iloc[-1], f"N0 = {n0:,}: s = {pct(s)}"))
    ax.plot([], [], color=MUTED, lw=1, ls=(0, (2, 2)), label="Gustafson fit s + (1 − s) P")
    style(ax, "MPI ranks P (problem grows with P)", "Scaled speed-up τ(P) / T(P), step loop", p_all)
    ax.set_ylim(0, max(p_all) * 1.05)
    ax.legend(loc="upper left")
    end_labels(ax, ends)
    save(fig, out, f"scaled_speedup_{tag}")

    # weak efficiency
    fig, ax = plt.subplots()
    ax.axhline(100, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal")
    top = 110.0
    for i, (n0, d) in enumerate(sweeps):
        c = SERIES[i % len(SERIES)]
        y = 100 * d["efficiency"]
        ax.errorbar(d["n_ranks"], y, yerr=[y - 100 * d["efficiency_lo"], 100 * d["efficiency_hi"] - y],
                    color=c, marker="o", ms=5, capsize=2, label=f"N0 = {n0:,}")
        top = max(top, float(100 * d["efficiency_hi"].max()) + 5)
    style(ax, "MPI ranks P (problem grows with P)", "Weak efficiency T(1) / T(P) (%), step loop", p_all)
    ax.set_ylim(0, top)
    ax.legend(loc="lower left")
    save(fig, out, f"weak_efficiency_{tag}")

    # serial fraction implied by every P
    fig, ax = plt.subplots()
    ax.axhline(0, color=MUTED, lw=1)
    for i, (n0, d) in enumerate(sweeps):
        c = SERIES[i % len(SERIES)]
        d = d[d["n_ranks"] > 1]
        ax.plot(d["n_ranks"], 100 * d["s_of_p"], color=c, marker="o", ms=5, label=f"N0 = {n0:,}")
        if np.isfinite(d["s_fit"].iloc[0]):
            ax.axhline(100 * d["s_fit"].iloc[0], color=c, lw=1, ls=(0, (2, 2)))
    ax.plot([], [], color=MUTED, lw=1, ls=(0, (2, 2)), label="fitted s")
    style(ax, "MPI ranks P", "Gustafson serial fraction s(P) = (P − S)/(P − 1) (%)", [p for p in p_all if p > 1])
    ax.legend(loc="best")
    save(fig, out, f"gustafson_fraction_{tag}")


def plot_growth(res: pd.DataFrame, label: str, out: Path) -> None:
    """tau(P)/tau(1) for every sweep: how much more serial work the scaled problem really is."""
    loop = res[(res["label"] == label) & (res["measure"] == "loop")]
    if loop.empty:
        return
    fig, ax = plt.subplots()
    p_all = sorted(int(p) for p in loop["n_ranks"].unique())
    ax.plot(p_all, p_all, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal: P times the work")
    i = 0
    for rule in ("time", "pairs"):
        for n0, d in latest(res, label, rule):
            ax.plot(d["n_ranks"], d["growth"], color=SERIES[i % len(SERIES)], marker="o" if rule == "time" else "s",
                    ms=5, ls="-" if rule == "time" else (0, (5, 2)), label=f"{rule}, N0 = {n0:,}")
            i += 1
    style(ax, "MPI ranks P", "Serial work of the scaled problem τ(P) / τ(1)", p_all)
    ax.set_ylim(0, 1.05 * max(max(p_all), float(loop["growth"].max())))
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, f"growth_{label}")


def plot_rules(res: pd.DataFrame, label: str, out: Path) -> None:
    """For every N0 that ran with both rules: weak efficiency and scaled speed-up side by side."""
    by = {rule: dict(latest(res, label, rule)) for rule in ("time", "pairs")}
    for n0 in sorted(set(by["time"]) & set(by["pairs"])):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
        p_all = sorted(int(p) for p in by["time"][n0]["n_ranks"])
        a1.axhline(100, ls=(0, (4, 4)), lw=1, color=MUTED)
        a2.plot(p_all, p_all, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal")
        for i, rule in enumerate(("time", "pairs")):
            d = by[rule][n0]
            a1.plot(d["n_ranks"], 100 * d["efficiency"], color=SERIES[i], marker="o", ms=5, label=RULE_TEXT[rule])
            a2.plot(d["n_ranks"], d["speedup"], color=SERIES[i], marker="o", ms=5, label=RULE_TEXT[rule])
        style(a1, "MPI ranks P", "Weak efficiency T(1) / T(P) (%)", p_all)
        style(a2, "MPI ranks P", "Scaled speed-up τ(P) / T(P)", p_all)
        a1.set_ylim(bottom=0)
        a2.set_ylim(0, max(p_all) * 1.05)
        a1.legend(loc="lower left", fontsize=8.5)
        a2.legend(loc="upper left", fontsize=8.5)
        fig.tight_layout()
        save(fig, out, f"rules_{label}_N0_{n0}")


def plot_measures(res: pd.DataFrame, tag: str, out: Path) -> None:
    fig, ax = plt.subplots()
    p_all = sorted(res["n_ranks"].unique())
    ax.axhline(100, ls=(0, (4, 4)), lw=1, color=MUTED, label="ideal")
    for i, m in enumerate(m for m in MEASURES if (res["measure"] == m).any()):
        d = res[res["measure"] == m].sort_values("n_ranks")
        ax.plot(d["n_ranks"], 100 * d["efficiency"], color=SERIES[i], marker="o", ms=5, label=MEASURE_TEXT[m])
    style(ax, "MPI ranks P (problem grows with P)", "Weak efficiency T(1) / T(P) (%)", p_all)
    ax.set_ylim(0, max(110, 100 * res["efficiency"].max() + 5))
    ax.legend(loc="lower left")
    save(fig, out, f"measures_{tag}")


def plot_lost(lost: pd.DataFrame, tag: str, out: Path) -> None:
    fig, ax = plt.subplots()
    xs = np.arange(len(lost))
    pos, neg = np.zeros(len(lost)), np.zeros(len(lost))
    for key, text, colour in LOSS_PARTS:
        v = lost[f"lost_{key}_pct"].to_numpy()
        ax.bar(xs, v, 0.62, bottom=np.where(v >= 0, pos, neg), color=colour, edgecolor="white", linewidth=1, label=text)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.plot(xs, lost["lost_total_pct"], ls="none", marker="_", ms=16, mew=2, color=INK, label="total = 1/E − 1")
    ax.axhline(0, color=MUTED, lw=1)
    style(ax, "MPI ranks P", "Extra time per step (% of T(1)), step loop", [])
    ax.set_xticks(xs, [str(p) for p in lost["n_ranks"]])
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, f"lost_time_{tag}")


# ------------------------------------------------------------------ tables and report
def num(v: float) -> str:
    if not np.isfinite(v):
        return "---"
    return f"${v:.0f}$" if abs(v) >= 100 else f"${v:.3g}$"


def latex_table(res: pd.DataFrame, info: Dict, out: Path, tag: str) -> None:
    d = res[res["measure"] == "loop"].sort_values("n_ranks")
    s = d["s_fit"].iloc[0]
    rule_text = r"every rank keeps the serial step time of $N_0$" if info["rule"] == "time" else r"$N = N_0\sqrt{P}$"
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{Weak scaling of \texttt{{{info['label']}}} from $N_0 = {info['n0']:,}$ stars "
             rf"({rule_text}), "
             rf"one node ({info['hostname']}, job {info['job']}), times per step of the step loop, medians over repeats. "
             r"$T$: MPI with $P$ ranks on $N$ stars; $\tau$: serial program on the same $N$; "
             r"$E = T(1)/T(P)$; $S = \tau/T$ (scaled speed-up); $s(P) = (P - S)/(P - 1)$. "
             + (rf"Gustafson fit $S = s + (1 - s)P$: $s = {100 * s:.3g}\,\%$.}}" if np.isfinite(s) else "}"),
             rf"  \label{{tab:gustafson_{tag}}}", r"  \begin{tabular}{rrrrrrr}", r"    \toprule",
             r"    $P$ & $N$ & $\tau$ [ms] & $T$ [ms] & $E$ & $S$ & $s(P)$ \\", r"    \midrule"]
    for _, r in d.iterrows():
        sp = "---" if r["n_ranks"] == 1 else f"{num(100 * r['s_of_p'])}\\,\\%"
        lines.append(rf"    {int(r['n_ranks'])} & {int(r['n_stars']):,} & {num(r['tau_ms'])} & {num(r['t_ms'])} & "
                     rf"{100 * r['efficiency']:.1f}\,\% & {r['speedup']:.2f} & {sp} \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / f"gustafson_table_{tag}.tex").write_text("\n".join(lines) + "\n")


def by_n0_table(res: pd.DataFrame, label: str, out: Path) -> pd.DataFrame | None:
    rows = []
    for rule in ("time", "pairs"):
        for n0, d in latest(res, label, rule):
            last = d.iloc[-1]
            kf = d[d["n_ranks"] > 1].set_index("n_ranks")["s_of_p"]
            p_first = next((p for p in kf.index if p >= 4), kf.index[0]) if len(kf) else np.nan
            rows.append({"label": label, "rule": rule, "n0": n0, "job": d["job"].iloc[0], "steps": int(d["steps"].iloc[0]),
                         "t1_ms": float(d["t1_ms"].iloc[0]), "p_max": int(last["n_ranks"]),
                         "n_pmax": int(last["n_stars"]), "t_pmax_ms": float(last["t_ms"]),
                         "tau_pmax_ms": float(last["tau_ms"]), "efficiency_pmax": float(last["efficiency"]),
                         "speedup_pmax": float(last["speedup"]), "scaled_efficiency_pmax": float(last["scaled_efficiency"]),
                         "growth_pmax": float(last["growth"]), "pairs_per_rank_pmax": float(last["pairs_per_rank"]),
                         "s_fit": float(d["s_fit"].iloc[0]), "s_lo": float(d["s_lo"].iloc[0]), "s_hi": float(d["s_hi"].iloc[0]),
                         "p_s_first": p_first, "s_first": kf.get(p_first, np.nan) if len(kf) else np.nan,
                         "s_pmax": kf.iloc[-1] if len(kf) else np.nan, "hostname": d["hostname"].iloc[0]})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    t.to_csv(out / f"by_N0_{label}.csv", index=False)
    pm = int(t["p_max"].max())
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{Weak scaling of the step loop of \texttt{{{label}}}: one fully reserved node per sweep, "
             rf"medians over repeats. Rule \emph{{time}}: $N(P)$ keeps the serial step time per rank; "
             r"\emph{pairs}: $N = N_0\sqrt{P}$. $T(1)$: serial time per step on $N_0$ stars. "
             r"$E = T(1)/T(P)$, $S = \tau(P)/T(P)$, $g = \tau(P)/\tau(1)$ (growth of the serial work, ideal $P$). "
             r"$s$: Gustafson fit $S = s + (1-s)P$, in brackets the fit to the fastest and the slowest runs.}",
             rf"  \label{{tab:gustafson_by_n0_{label}}}", r"  \begin{tabular}{llrrrrrrrr}", r"    \toprule",
             rf"    rule & $N_0$ & $N({pm})$ & $T(1)$ [ms] & $E({pm})$ & $S({pm})$ & $g({pm})$ & $s$ & (runs) & "
             rf"$s(4) \to s({pm})$ \\", r"    \midrule"]
    for _, r in t.iterrows():
        rng = rf"({num(100 * r['s_lo'])} to {num(100 * r['s_hi'])}\,\%)" if np.isfinite(r["s_lo"]) else "---"
        lines.append(rf"    {r['rule']} & {r['n0']:,} & {r['n_pmax']:,} & {num(r['t1_ms'])} & "
                     rf"{100 * r['efficiency_pmax']:.0f}\,\% & {r['speedup_pmax']:.1f} & {r['growth_pmax']:.1f} & "
                     rf"{num(100 * r['s_fit'])}\,\% & {rng} & {num(100 * r['s_first'])}\,\% $\to$ {num(100 * r['s_pmax'])}\,\% \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / f"by_N0_{label}.tex").write_text("\n".join(lines) + "\n")
    return t


def print_report(res: pd.DataFrame, info: Dict) -> None:
    d = res[res["measure"] == "loop"].sort_values("n_ranks")
    print(f"\n=== {info['label']}, rule {info['rule']} ({RULE_TEXT[info['rule']]}), N0 = {info['n0']:,}, "
          f"{int(d['steps'].iloc[0]):,} steps, job {info['job']} on {info['hostname']}")
    print(f"  {'P':>3} {'N':>6} {'serial steps':>12} {'tau [ms]':>10} {'T [ms]':>9} {'E':>7} {'S':>7} "
          f"{'S/P':>6} {'growth':>7} {'pairs/rank':>10} {'s(P)':>8}  repeats")
    for _, r in d.iterrows():
        sp = "" if r["n_ranks"] == 1 else pct(r["s_of_p"])
        print(f"  {int(r['n_ranks']):>3} {int(r['n_stars']):>6,} {r['serial_steps']:>12.0f} {r['tau_ms']:>10.4g} "
              f"{r['t_ms']:>9.4g} {100 * r['efficiency']:>6.1f}% {r['speedup']:>7.2f} {100 * r['scaled_efficiency']:>5.0f}% "
              f"{r['growth']:>7.2f} {r['pairs_per_rank']:>9.2f}× {sp:>8}  {int(r['repeats'])}/{int(r['serial_repeats'])}")
    s = d["s_fit"].iloc[0]
    rng = f"   (runs: {pct(d['s_lo'].iloc[0])} .. {pct(d['s_hi'].iloc[0])})" if np.isfinite(d["s_lo"].iloc[0]) else ""
    print(f"  Gustafson fit S = s + (1 - s) P:  s = {pct(s)}{rng}")
    for m in ("program", "launch"):
        w = res[res["measure"] == m].sort_values("n_ranks")
        if len(w):
            print(f"  weak efficiency of the {MEASURE_TEXT[m]}: " +
                  "  ".join(f"P={int(r['n_ranks'])}: {100 * r['efficiency']:.0f}%" for _, r in w.iterrows()))
    p1 = d[d["n_ranks"] == 1]
    if len(p1) and abs(p1["efficiency"].iloc[0] - 1) > 0.02:
        e = p1["efficiency"].iloc[0]
        print(f"  the MPI program at P = 1 is {abs(e - 1):.0%} {'faster' if e > 1 else 'slower'} than the serial "
              f"program; E and S are relative to the serial program")
    below = d[d["efficiency"] < 0.9]["n_ranks"]
    print(f"  weak efficiency drops below 90 % at P = {int(below.iloc[0])}" if len(below)
          else "  weak efficiency stays at 90 % or more for all P")


def startup(df: pd.DataFrame, label: str, out: Path) -> pd.DataFrame | None:
    d = df[(df["label"] == label) & df["launch_wall_s"].notna()].copy()
    if d.empty:
        return None
    d["startup_s"] = d["launch_wall_s"] - d["total_time_s"]
    t = (d.groupby(["rule", "n0", "implementation", "n_ranks", "n_stars_start"])["startup_s"]
         .agg(["median", "min", "max", "count"]).reset_index())
    t.insert(0, "label", label)
    t.to_csv(out / f"startup_{label}.csv", index=False)
    mpi = t[t["implementation"] == "mpi"]
    if mpi["n_ranks"].nunique() > 1:
        b, a = np.polyfit(mpi["n_ranks"], mpi["median"], 1)
        ser = t[t["implementation"] == "serial"]["median"]
        print(f"\n=== {label}: start-up per launch (launch wall time - the program's own time)")
        print(f"  mpirun launches: {a:.2f} s + {1000 * b:.0f} ms per rank; serial program {ser.min():.2f}-{ser.max():.2f} s")
    return t


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
            print(f"Skipping {dict(zip(GROUP, key))}: needs the serial run on N0 and MPI runs.")
            continue
        info = dict(zip(GROUP, key)) | {"hostname": g["hostname"].iloc[0]}
        for col in GROUP + ["hostname"]:
            res.insert(0 if col != "hostname" else len(res.columns), col, info[col])
        tag = f"{info['label']}_{info['rule']}_N0_{info['n0']}"
        same = df[(df["label"] == info["label"]) & (df["rule"] == info["rule"]) & (df["n0"] == info["n0"])]
        if same["job"].nunique() > 1:
            tag += f"_j{info['job']}"
        print_report(res, info)
        lost = lost_time(g, res)
        for i, col in enumerate(GROUP):
            lost.insert(i, col, info[col])
        all_lost.append(lost)
        plot_lost(lost, tag, args.out)
        plot_measures(res, tag, args.out)
        latex_table(res, info, args.out, tag)
        all_res.append(res)
    if not all_res:
        raise SystemExit("No complete sweep found.")
    res = pd.concat(all_res, ignore_index=True)
    for label in sorted(res["label"].unique()):
        for rule in ("time", "pairs"):
            plot_rule(res, label, rule, args.out)
        plot_growth(res, label, args.out)
        plot_rules(res, label, args.out)
        t = by_n0_table(res, label, args.out)
        if t is not None:
            print(f"\n=== {label}: every sweep at the largest P (latest job per rule and N0)")
            print(f"  {'rule':<6} {'N0':>6} {'N(Pmax)':>8} {'T(1) ms':>9} {'E':>6} {'S':>6} {'growth':>7} "
                  f"{'Gustafson s':>12} {'s from all runs':>18}   s(P)")
            for _, r in t.iterrows():
                rng = f"{pct(r['s_lo'])} .. {pct(r['s_hi'])}" if np.isfinite(r["s_lo"]) else "n/a"
                print(f"  {r['rule']:<6} {r['n0']:>6,} {r['n_pmax']:>8,} {r['t1_ms']:>9.4g} "
                      f"{100 * r['efficiency_pmax']:>5.0f}% {r['speedup_pmax']:>6.1f} {r['growth_pmax']:>7.1f} "
                      f"{pct(r['s_fit']):>12} {rng:>18}   {pct(r['s_first'])} (P = {r['p_s_first']:.0f}) -> "
                      f"{pct(r['s_pmax'])} (P = {r['p_max']})")
            print("  growth = tau(Pmax)/tau(1): how much more serial work the scaled problem is (ideal Pmax). S = growth * E.")
        startup(df, label, args.out)
    res.to_csv(args.out / "gustafson_summary.csv", index=False)
    pd.concat(all_lost, ignore_index=True).to_csv(args.out / "lost_time_summary.csv", index=False)
    print(f"\nWrote plots, LaTeX tables, gustafson_summary.csv and lost_time_summary.csv to {args.out}")


if __name__ == "__main__":
    main()
