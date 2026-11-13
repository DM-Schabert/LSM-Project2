"""Analyse C0: latency and bandwidth of the cluster, and what each scheme's exchange costs.

Reads every communication/results/bench_*.csv (or the files given) and writes to
communication/out/bench/:

    pingpong.png/.pdf    one-way time against message size per pair of ranks, with the fit T = mu + s/B
    schemes_N.png/.pdf   time of one exchange against N at the largest P, per scheme, one panel per job
    schemes_P.png/.pdf   time of one exchange against P for a small and a large N, per job
    fits.csv             mu and B per job and pair
    report.txt           what is printed

The ring is also predicted from the pingpong fit, (P - 1) (mu + b/B) with b the block size and
mu, B of the slowest link it uses (the other node if the ring spans two nodes): the expectation
the measurement is compared with.

Usage (from the repository root, needs pandas and matplotlib):
    python3 communication/bench/analyse_bench.py [files ...] [--out communication/out/bench]
"""
import argparse
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
COMM = HERE.parent
SCHEMES = [("allgatherv", "Allgatherv (v1)", "#2a78d6"), ("ring", "ring, one turn", "#1baf7a"),
           ("2d", "2D, one step", "#6250d6")]
PAIRS = {"neighbouring core": "#2a78d6", "other half of the node": "#1baf7a", "other node": "#6250d6"}
INK, MUTED, GRID = "#121823", "#6f798b", "#dce1e8"
plt.rcParams.update({"font.size": 10, "legend.frameon": False, "figure.figsize": (6.2, 4.0), "lines.linewidth": 1.8})


def style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def job_title(d: pd.DataFrame) -> str:
    n = int(d["n_nodes"].iloc[0])
    return f"job {d['job'].iloc[0]}, {n} node{'s' if n > 1 else ''}"


def hockney(s: np.ndarray, t: np.ndarray) -> tuple:
    """Fit t = mu + s/B with relative errors (weights 1/t), so small and large messages count alike."""
    a = np.column_stack([1 / t, s / t])
    (mu, beta), *_ = np.linalg.lstsq(a, np.ones_like(t), rcond=None)
    return mu, (1 / beta if beta > 0 else np.inf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=COMM / "out" / "bench")
    args = ap.parse_args()
    files: List[Path] = args.files or sorted((COMM / "results").glob("bench_*.csv"))
    if not files:
        raise SystemExit("No bench_*.csv in communication/results/. Copy them from the cluster first.")
    df = pd.concat([pd.read_csv(f, dtype={"job": str}) for f in files], ignore_index=True)
    df["t"] = df["t_median_us"] * 1e-6
    args.out.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731

    # ------------------------------------------------------------ latency and bandwidth
    pp = df[df["pattern"] == "pingpong"]
    fits = []
    for (job, pair), g in pp.groupby(["job", "pair"]):
        g = g.sort_values("message_bytes")
        mu, bw = hockney(g["message_bytes"].to_numpy(float), g["t"].to_numpy())
        fits.append({"job": job, "n_nodes": int(g["n_nodes"].iloc[0]), "pair": pair, "mu_us": mu * 1e6,
                     "bandwidth_GBps": bw / 1e9, "t_8B_us": float(g["t_median_us"].iloc[0]),
                     "hosts": g["hosts"].iloc[0], "mpi_library": g["mpi_library"].iloc[0]})
    fits = pd.DataFrame(fits)
    if len(fits):
        fits.to_csv(args.out / "fits.csv", index=False)
        say("=== Latency and bandwidth (pingpong, one way; fit T = mu + s/B)")
        for _, f in fits.iterrows():
            say(f"  job {f['job']:<20} {f['pair']:<24} mu = {f['mu_us']:6.2f} us   B = {f['bandwidth_GBps']:6.2f} GB/s"
                f"   (8 B message: {f['t_8B_us']:.2f} us)")
        jobs = list(dict.fromkeys(pp["job"]))
        fig, axes = plt.subplots(1, len(jobs), figsize=(6.2 * len(jobs), 4.0), squeeze=False)
        for ax, job in zip(axes[0], jobs):
            g = pp[pp["job"] == job]
            for pair, h in g.groupby("pair"):
                h = h.sort_values("message_bytes")
                c = PAIRS.get(pair, MUTED)
                ax.errorbar(h["message_bytes"], h["t_median_us"],
                            yerr=[h["t_median_us"] - h["t_p25_us"], h["t_p75_us"] - h["t_median_us"]],
                            color=c, marker="o", ms=4, capsize=2, label=pair)
                f = fits[(fits["job"] == job) & (fits["pair"] == pair)].iloc[0]
                s = np.logspace(np.log10(h["message_bytes"].min()), np.log10(h["message_bytes"].max()), 100)
                ax.plot(s, (f["mu_us"] * 1e-6 + s / (f["bandwidth_GBps"] * 1e9)) * 1e6, color=c, lw=1, ls=(0, (2, 2)))
            ax.set_xscale("log")
            ax.set_yscale("log")
            style(ax, "message size (bytes)", "one-way time (µs)")
            ax.set_title(job_title(g), color=INK, fontsize=10)
            ax.plot([], [], color=MUTED, lw=1, ls=(0, (2, 2)), label="fit μ + s/B")
            ax.legend(loc="upper left")
        save(fig, args.out, "pingpong")

    # ------------------------------------------------------------ the schemes' exchanges
    sch = df[df["pattern"].isin([s for s, _, _ in SCHEMES])].copy()
    sch["n_stars"] = sch["n_stars"].astype(int)
    jobs = list(dict.fromkeys(sch["job"]))
    if jobs:
        fig, axes = plt.subplots(1, len(jobs), figsize=(6.2 * len(jobs), 4.0), squeeze=False)
        for ax, job in zip(axes[0], jobs):
            g = sch[sch["job"] == job]
            pmax = int(g["n_ranks"].max())
            n_nodes = int(g["n_nodes"].iloc[0])
            say(f"\n=== Exchange per time step at P = {pmax}, {job_title(g)} (median, µs)")
            say(f"  {'N':>8} {'Allgatherv':>11} {'ring':>11} {'2D':>11} {'ring model':>11}")
            link = "other node" if n_nodes > 1 else "neighbouring core"
            f = fits[(fits["job"] == job) & (fits["pair"] == link)] if len(fits) else pd.DataFrame()
            table = g[g["n_ranks"] == pmax].pivot_table(index="n_stars", columns="pattern", values="t_median_us")
            ring_b = g[(g["n_ranks"] == pmax) & (g["pattern"] == "ring")].set_index("n_stars")["message_bytes"]
            model = ((pmax - 1) * (f["mu_us"].iloc[0] + ring_b / (f["bandwidth_GBps"].iloc[0] * 1e3))
                     if len(f) else pd.Series(dtype=float))
            for n, r in table.iterrows():
                cells = [f"{r.get(p, np.nan):11.1f}" for p in ("allgatherv", "ring", "2d")]
                say(f"  {n:>8,} " + " ".join(cells) + f" {model.get(n, np.nan):11.1f}")
            for pattern, label, c in SCHEMES:
                h = g[(g["pattern"] == pattern) & (g["n_ranks"] == pmax)].sort_values("n_stars")
                if len(h):
                    ax.errorbar(h["n_stars"], h["t_median_us"],
                                yerr=[h["t_median_us"] - h["t_p25_us"], h["t_p75_us"] - h["t_median_us"]],
                                color=c, marker="o", ms=4, capsize=2, label=label)
            if len(model):
                ax.plot(model.index, model.values, color="#1baf7a", lw=1, ls=(0, (2, 2)), label="ring, model from pingpong")
            ax.set_xscale("log")
            ax.set_yscale("log")
            style(ax, "stars N", f"time per exchange at P = {pmax} (µs)")
            ax.set_title(job_title(g), color=INK, fontsize=10)
            ax.legend(loc="upper left")
        save(fig, args.out, "schemes_N")

        fig, axes = plt.subplots(len(jobs), 2, figsize=(12.4, 4.0 * len(jobs)), squeeze=False)
        for row, job in zip(axes, jobs):
            g = sch[sch["job"] == job]
            ns = sorted(g["n_stars"].unique())
            for ax, n in zip(row, [min((x for x in ns if x >= 1000), default=ns[0]), ns[-1]]):
                for pattern, label, c in SCHEMES:
                    h = g[(g["pattern"] == pattern) & (g["n_stars"] == n)].sort_values("n_ranks")
                    if len(h):
                        ax.plot(h["n_ranks"], h["t_median_us"], color=c, marker="o", ms=4, label=label)
                ax.set_yscale("log")
                ax.set_xticks(sorted(g["n_ranks"].unique()))
                style(ax, "ranks P", "time per exchange (µs)")
                ax.set_title(f"{job_title(g)}, N = {n:,}", color=INK, fontsize=10)
                ax.legend(loc="upper left")
        save(fig, args.out, "schemes_P")

    ar = df[df["pattern"] == "allreduce"]
    if len(ar):
        say("\n=== Allreduce of one integer (a collective without data), median µs")
        for job, g in ar.groupby("job"):
            say(f"  {job_title(g)}: " + "   ".join(f"P = {int(r.n_ranks)}: {r.t_median_us:.1f}"
                                                 for r in g.sort_values("n_ranks").itertuples()))
    (args.out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}/: pingpong, schemes_N, schemes_P (PNG + PDF), fits.csv, report.txt")


if __name__ == "__main__":
    main()
