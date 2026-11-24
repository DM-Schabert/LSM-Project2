"""bench5: how many time steps of each group of stars fit into 5 minutes, for every configuration of the project.

    python3 bench5/analyze.py            # the cluster jobs in bench5/out/ (files of LOCAL=1 runs are left out)
    python3 bench5/analyze.py --local    # only the LOCAL=1 files: checks that the scripts and this analysis run

Reads bench5/out/runs_<job>.csv (phase 1: bench5_<config>_N<N>_r<rep>_j<job>; phase 2: bench5final_<config>_N<N>_j<job>).
Writes bench5/out/steps5.csv (every N and configuration), winners.txt (the best configuration per N, the input of
bench5_final.sh), report.txt (the expectations of README.md with their verdicts) and plots/steps5.png.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LIMIT_S, RESERVE = 300.0, 0.05      # 5 minutes; the program stops 5 % before its time limit
PHASE1 = re.compile(r"^bench5_(?P<label>[\w-]+?)_N(?P<N>\d+)_r(?P<rep>\d+)_j(?P<job>\w+)$")
PHASE2 = re.compile(r"^bench5final_(?P<label>[\w-]+?)_N(?P<N>\d+)_j(?P<job>\w+)$")


def configs() -> pd.DataFrame:
    rows = []
    for line in (HERE / "configs.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            f = [x.strip() for x in line.split("|")]
            rows.append({"label": f[0], "ranks": int(f[1]), "placement": f[2], "scheme": f[3], "tiles": f[4], "since": f[5], "what": f[6]})
    c = pd.DataFrame(rows)
    c["order"] = range(len(c))
    return c


def runs(local: bool) -> pd.DataFrame:
    fs = sorted(p for p in OUT.glob("runs_*.csv") if ("local" in p.name) == local)
    return pd.concat([pd.read_csv(p) for p in fs], ignore_index=True) if fs else pd.DataFrame(columns=["run_name"])


def measure(d: pd.DataFrame) -> tuple:
    """Phase 1 per (N, label): median time per step, steps in 5 minutes of computing and in a 5-minute job."""
    rows, final = [], []
    for r in d.itertuples():
        m1, m2 = PHASE1.match(str(r.run_name)), PHASE2.match(str(r.run_name))
        it = int(r.iterations) if pd.notna(r.iterations) else 0
        if it < 1:
            continue
        step = r.simulation_time_s / it
        startup = max(r.total_time_s - r.simulation_time_s, 0.0)
        if m1:
            rows.append({"N": int(m1["N"]), "label": m1["label"], "rep": int(m1["rep"]), "step_s": step, "startup_s": startup,
                         "ranks": int(r.n_ranks) if pd.notna(r.n_ranks) else 1, "tile_rows": r.kernel_tile_rows if "kernel_tile_rows" in d else np.nan})
        elif m2:
            final.append({"N": int(m2["N"]), "label": m2["label"], "steps_in_job": it, "step_s_final": step, "startup_s_final": startup,
                          "stop_reason": r.stop_reason, "total_s": r.total_time_s, "limit_s": r.time_limit_s})
    p1 = pd.DataFrame(rows)
    if p1.empty:
        return p1, pd.DataFrame(final)
    g = p1.groupby(["N", "label"], as_index=False).agg(repeats=("rep", "size"), step_s=("step_s", "median"), step_min=("step_s", "min"),
                                                       step_max=("step_s", "max"), startup_s=("startup_s", "median"), tile_rows=("tile_rows", "first"))
    g["steps_5min"] = LIMIT_S / g["step_s"]                                       # 5 minutes of computing
    g["steps_5min_job"] = (LIMIT_S * (1 - RESERVE) - g["startup_s"]) / g["step_s"]   # a 5-minute job: start-up and reserve off
    return g, pd.DataFrame(final)


def table(g: pd.DataFrame, fin: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    t = g.merge(c, on="label", how="left").sort_values(["N", "order"])
    serial = t[t.label == "serial"].set_index("N").steps_5min
    t["x_serial"] = t.apply(lambda r: r.steps_5min / serial[r.N] if r.N in serial.index else np.nan, axis=1)
    t["best_so_far"] = t.groupby("N").steps_5min.cummax()                      # in the order the project found the configurations
    t["rank_in_N"] = t.groupby("N").steps_5min.rank(ascending=False, method="min").astype(int)
    if not fin.empty:
        t = t.merge(fin[["N", "label", "steps_in_job", "stop_reason", "limit_s"]], on=["N", "label"], how="left")
        t["steps_expected"] = (t["limit_s"] * (1 - RESERVE) - t["startup_s"]) / t["step_s"]   # for the run's own time limit
    return t


def verdicts(t: pd.DataFrame) -> list:
    res = []

    def add(eid, text, fn):
        try:
            v, detail = fn()
        except (KeyError, IndexError, ValueError):
            v, detail = None, "no data"
        res.append((eid, text, "n.a." if v is None else ("yes" if v else "no"), detail))

    def at(n, label):
        r = t[(t.N == n) & (t.label == label)]
        if r.empty:
            raise KeyError(label)
        return float(r.steps_5min.iloc[0])

    win = t.loc[t.groupby("N").steps_5min.idxmax()].set_index("N")
    big = [n for n in win.index if n >= 5000]
    add("B1", "From N = 5 000 on the best configuration uses two nodes.",
        lambda: (all(win.loc[n, "ranks"] == 48 for n in big) if big else None, "; ".join(f"N={n}: {win.loc[n, 'label']}" for n in big)))

    def b2():
        ns = [n for n in sorted(t.N.unique()) if n <= 20000]
        pairs = [(n, at(n, "v1-small-24"), at(n, "v1-tiles-24")) for n in ns]
        return all(a > b for _, a, b in pairs), "; ".join(f"N={n}: {a:.0f} vs {b:.0f} steps" for n, a, b in pairs)
    add("B2", "Up to N = 20 000, v1 with 2-row tiles beats the automatic tiles on one node.", b2)

    def b3():
        v1 = max(at(1000, x) for x in ("v1-tiles-24", "v1-tiles-48", "v1-small-24", "v1-small-48"))
        ring = max(at(1000, x) for x in ("ring-24", "ring-48", "ring-small-24", "ring-small-48"))
        return v1 > ring, f"best v1 {v1:.0f}, best ring {ring:.0f} steps"
    add("B3", "At N = 1 000 v1 beats every ring (the ring's P kernel calls per step).", b3)

    def b4():
        w = win.loc[100000, "label"]
        return w.startswith("ring-small"), f"winner {w}"
    add("B4", "At N = 100 000 the ring with ~1 MB pieces wins.", b4)

    def b5():
        f = t.dropna(subset=["steps_in_job"]) if "steps_in_job" in t else t.iloc[0:0]
        if f.empty:
            raise KeyError("phase 2")
        q = f.steps_in_job / f.steps_expected
        return bool(((q >= 0.9) & (q <= 1.05)).all()), "; ".join(f"N={r.N}: {int(r.steps_in_job)} of {r.steps_expected:.0f} expected" for r in f.itertuples())
    add("B5", "A real 5-minute run reaches 90-105 % of the steps estimated from phase 1 (start-up included).", b5)
    return res


def plot(t: pd.DataFrame, where: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted(t.N.unique())
    fig, axes = plt.subplots(len(ns), 1, figsize=(8, 2.4 + 0.32 * len(t) / max(len(ns), 1) * len(ns)), squeeze=False)
    color = {"serial": "#9aa1ac", "allgather": "#2a78d6", "ring": "#eb6834", "2d": "#1baf7a"}
    for ax, n in zip(axes[:, 0], ns):
        s = t[t.N == n].sort_values("order")
        best = s.steps_5min.max()
        y = np.arange(len(s))
        from matplotlib.colors import to_rgba
        ax.barh(y, s.steps_5min, color=[to_rgba(color.get(x, "#888"), 1.0 if v == best else 0.5) for x, v in zip(s.scheme, s.steps_5min)])
        for yi, r in zip(y, s.itertuples()):
            ax.text(r.steps_5min * 1.03, yi, f"{r.steps_5min:,.0f}".replace(",", " ") + (f"  (5-min run: {int(r.steps_in_job):,})".replace(",", " ")
                    if "steps_in_job" in s and pd.notna(getattr(r, "steps_in_job", np.nan)) else ""), va="center", fontsize=7.5)
        ax.set_yticks(y, [f"{r.label}  ({r.since})" for r in s.itertuples()], fontsize=7.5)
        ax.invert_yaxis(); ax.set_xscale("log"); ax.grid(axis="x", alpha=.3)
        ax.set_title(f"N = {n:,}".replace(",", " ") + ": time steps in 5 minutes of computing", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("time steps in 5 minutes (log)")
    fig.tight_layout()
    where.mkdir(parents=True, exist_ok=True)
    fig.savefig(where / "steps5.png", dpi=150)
    plt.close(fig)
    return "steps5.png"


def plot_progress(t: pd.DataFrame, c: pd.DataFrame, where: Path) -> str:
    """All tests on one plot: every configuration in the order the project built it, one line per group of stars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = c.sort_values("order").reset_index(drop=True)
    x_of = {lab: i for i, lab in enumerate(cfg.label)}
    ns = sorted(t.N.unique())
    blues = ["#86b6ef", "#4a8fe0", "#1c5cab", "#0c3366"][-len(ns):] if len(ns) <= 4 else plt.cm.Blues(np.linspace(0.4, 1, len(ns)))
    fig, ax = plt.subplots(figsize=(13.5, 6))
    for i, lab in enumerate(cfg.label):                                   # two-node configurations on a grey band
        if int(cfg.loc[i, "ranks"]) > 24:
            ax.axvspan(i - 0.5, i + 0.5, color="#eceef2", zorder=0)
    for n, col in zip(ns, blues):
        s = t[t.N == n].assign(x=lambda d: d.label.map(x_of)).sort_values("x")
        ax.plot(s.x, s.steps_5min, "-o", color=col, lw=2, ms=5, zorder=3, label=f"N = {n:,}".replace(",", " "))
        w = s.loc[s.steps_5min.idxmax()]
        ax.plot(w.x, w.steps_5min, "o", ms=11, mfc="none", mec=col, mew=2, zorder=4)
        if "steps_in_job" in s and pd.notna(w.get("steps_in_job", np.nan)):
            ax.plot(w.x, w.steps_in_job, "*", ms=14, color=col, mec="#15181e", mew=0.6, zorder=5)
        ser = s[s.label == "serial"]
        gain = f"  ×{w.steps_5min / float(ser.steps_5min.iloc[0]):.0f} vs serial" if not ser.empty else ""
        ax.annotate(f"{w.label}: {w.steps_5min:,.0f}{gain}".replace(",", " "), (w.x, w.steps_5min), xytext=(6, 8),
                    textcoords="offset points", fontsize=8, color=col, zorder=6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cfg)), [f"{r.label}\n{r.since}" for r in cfg.itertuples()], rotation=40, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(cfg) - 0.4)
    ax.set_ylabel("time steps in 5 minutes (log), higher is better")
    ax.set_title("Every configuration we built, in order: how many time steps of each group of stars fit into 5 minutes", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.3, which="both")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    extra = [Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec="#5d6573", mew=2, label="best of the group"),
             Line2D([], [], marker="*", ls="", ms=12, color="#5d6573", label="real 5-minute run (phase 2)"),
             Patch(color="#eceef2", label="two nodes, 48 ranks")]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + extra, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    where.mkdir(parents=True, exist_ok=True)
    fig.savefig(where / "progress.png", dpi=150)
    plt.close(fig)
    return "progress.png"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", action="store_true", help="analyse the LOCAL=1 files only")
    args = ap.parse_args()
    tag = "_local" if args.local else ""
    g, fin = measure(runs(args.local))
    if g.empty:
        print("No phase-1 runs in bench5/out/ yet.")
        return
    t = table(g, fin, configs())
    t.to_csv(OUT / f"steps5{tag}.csv", index=False)
    win = t.loc[t.groupby("N").steps_5min.idxmax()]
    (OUT / f"winners{tag}.txt").write_text("# N and the configuration with the most time steps in 5 minutes (phase 1)\n"
                                         + "".join(f"{r.N} {r.label}\n" for r in win.itertuples()))
    made = plot(t, OUT / f"plots{tag}")
    made_all = plot_progress(t, configs(), OUT / f"plots{tag}")
    lines = [f"bench5: {len(g)} settings from phase 1, {len(fin)} runs of phase 2" + (" (LOCAL files)" if args.local else ""), ""]
    for n in sorted(t.N.unique()):
        s = t[t.N == n].sort_values("order")
        lines.append(f"N = {n}")
        for r in s.itertuples():
            fin_txt = f"   5-min run: {int(r.steps_in_job)} steps" if "steps_in_job" in s and pd.notna(getattr(r, "steps_in_job", np.nan)) else ""
            ser = (f"  x{r.x_serial:.1f}" if r.x_serial < 10 else f"  x{r.x_serial:.0f}") if pd.notna(r.x_serial) else ""
            lines.append(f"  {r.label:14s} {r.since:7s} {r.ranks:3d} ranks  {r.step_s * 1e3:10.1f} ms/step  {r.steps_5min:10.0f} steps in 5 min{ser}"
                         f"  ({r.steps_5min_job:.0f} in a 5-min job){'   <- best' if r.rank_in_N == 1 else ''}{fin_txt}")
        lines.append("")
    lines += [f"{eid}  {v:4s}  {text}\n             {detail}" for eid, text, v, detail in verdicts(t)]
    lines += ["", f"winners: out/winners{tag}.txt   table: out/steps5{tag}.csv   plots: out/plots{tag}/{made}, out/plots{tag}/{made_all} (all tests on one plot)"]
    (OUT / f"report{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
