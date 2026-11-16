"""Plots for isoeff campaigns (PNG for viewing, PDF for LaTeX; no titles, the file name is the title).

    python3 isoeff/plot.py isoeff/out/v1-numpy                      # one campaign
    python3 isoeff/plot.py isoeff/out/v1-numpy isoeff/out/v2-ring   # + version comparison
    python3 isoeff/plot.py isoeff/out/v* --out isoeff/plots

Per campaign:
    isoeff_<label>_speedup_vs_ranks      S(P), one line per N (Amdahl curves); hollow marker = threshold
    isoeff_<label>_efficiency_vs_ranks   E(P) = S/P, one line per N, with the E threshold
Over all capacity-mode campaigns given:
    capacity_<labels>_largest_N           the largest N that finishes the fixed steps in time, per P
    capacity_<labels>_marginal_efficiency the marginal efficiency of each step up in P
Over all time/step-mode campaigns given:
    isoeff_<labels>_isoefficiency        for each P, the N at which stepping up to P became
                                         cost-effective (hollow = step measured and not worth it):
                                         how fast N must grow to keep adding ranks (the Gustafson side)
Reads points.csv / thresholds.csv / config.json / DONE, which controller.py writes.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (dataviz skill): ordinal blue ramp for N (MPI is blue in this project),
# categorical order for versions, neutral grey for reference lines.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"
INK = "#52514e"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def ramp(k: int) -> List[str]:
    """k ordered colours from the blue ramp, light to dark, spread over its length."""
    if k <= 1:
        return [BLUE_RAMP[2]]
    idx = [round(i * (len(BLUE_RAMP) - 1) / (k - 1)) for i in range(k)]
    return [BLUE_RAMP[i] for i in idx]


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def legend(ax) -> None:
    """Legend to the right of the plot, so it never covers data."""
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.02, 1.0))


def save(fig, out_dir: Path, stem: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=160, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def per_n(points: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    lines: Dict[int, List[Dict[str, str]]] = {}
    for row in points:
        lines.setdefault(int(row["N"]), []).append(row)
    for rows in lines.values():
        rows.sort(key=lambda r: int(r["P"]))
    return dict(sorted(lines.items()))


def campaign_plots(camp: Path, out_dir: Path) -> List[Path]:
    cfg = json.loads((camp / "config.json").read_text())
    label = cfg["label"]
    points = read_csv(camp / "points.csv")
    if not points:
        return []
    lines = per_n(points)
    colours = ramp(len(lines))
    p_all = sorted({int(r["P"]) for r in points})
    saved: List[Path] = []

    for metric, ylabel, stem in (("speedup", "Speed-up S = T_serial / T_MPI", "speedup_vs_ranks"),
                                 ("efficiency", "Parallel efficiency E = S / P", "efficiency_vs_ranks")):
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        if metric == "speedup":
            ax.plot([1, max(p_all)], [1, max(p_all)], color=GREY, linestyle="--", linewidth=1,
                    label="ideal")
        else:
            ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1, label="ideal")
            ax.axhline(float(cfg["efficiency"]), color=GREY, linestyle=":", linewidth=1,
                       label=f"E threshold ({float(cfg['efficiency']):.0%})")
        for colour, (n, rows) in zip(colours, lines.items()):
            ps = [int(r["P"]) for r in rows]
            ys = [float(r[metric]) for r in rows]
            ax.plot(ps, ys, color=colour, linewidth=2, marker="o", markersize=5, label=f"N = {n:,}")
            for r, y in zip(rows, ys):
                if r["verdict"] == "threshold":  # the step that was not cost-effective
                    ax.plot(int(r["P"]), y, linestyle="", marker="o", markersize=10,
                            markerfacecolor="none", markeredgecolor=colour, markeredgewidth=1.5)
        ax.set_xlabel("MPI ranks P")
        ax.set_ylabel(ylabel)
        ax.set_xticks(p_all)
        if metric == "efficiency":
            ax.set_ylim(0, max(1.1, max(float(r["efficiency"]) for r in points) * 1.05))
        style(ax)
        legend(ax)
        saved += save(fig, out_dir, f"isoeff_{label}_{stem}")
    return saved


def isoefficiency_plot(camps: Sequence[Path], out_dir: Path) -> List[Path]:
    """For every P: the smallest N at which the step up to P was cost-effective.

    That is the isoefficiency path: how fast N has to grow for each extra step in P
    to stay worth it. Hollow markers are steps that were measured and not cost-effective.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    labels: List[str] = []
    p_seen: set = set()
    anything = False
    for i, camp in enumerate(camps):
        cfg = json.loads((camp / "config.json").read_text())
        labels.append(cfg["label"])
        colour = CATEGORICAL[i % len(CATEGORICAL)]
        points = read_csv(camp / "points.csv")
        reached: Dict[int, int] = {}
        for r in points:
            if r["verdict"] == "ok":
                p, n = int(r["P"]), int(r["N"])
                reached[p] = min(n, reached.get(p, n))
        failed = [(int(r["P"]), int(r["N"])) for r in points if r["verdict"] == "threshold"]
        if reached:
            xs = sorted(reached)
            ax.plot(xs, [reached[p] for p in xs], color=colour, linewidth=2, marker="o", markersize=6,
                    label=f"{cfg['label']}: N at which P became cost-effective")
            p_seen.update(xs)
            anything = True
        if failed:
            ax.plot([p for p, _ in failed], [n for _, n in failed], linestyle="", marker="o",
                    markersize=8, markerfacecolor="none", markeredgecolor=colour, markeredgewidth=1.5,
                    label=f"{cfg['label']}: step to P not cost-effective")
            p_seen.update(p for p, _ in failed)
            anything = True
        done = (camp / "DONE").read_text() if (camp / "DONE").exists() else ""
        if points and "still cost-effective" in done:
            last_n = max(int(r["N"]) for r in points)
            p_max = max(int(r["P"]) for r in points if int(r["N"]) == last_n)
            ax.plot([p_max], [last_n], linestyle="", marker="^", markersize=9, color=colour,
                    label=f"{cfg['label']}: still cost-effective at the largest P")
    if not anything:
        plt.close(fig)
        return []
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = sorted(p_seen)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("MPI ranks P")
    ax.set_ylabel("Stars N")
    style(ax)
    legend(ax)
    return save(fig, out_dir, f"isoeff_{'_vs_'.join(labels)}_isoefficiency")


def capacity_plots(camps: Sequence[Path], out_dir: Path) -> List[Path]:
    """Capacity mode: the largest N per P that finishes in time, and the marginal efficiency of each step."""
    camps = [c for c in camps if (c / "capacity.csv").exists()]
    if not camps:
        return []
    labels = [json.loads((c / "config.json").read_text())["label"] for c in camps]
    cfg0 = json.loads((camps[0] / "config.json").read_text())
    budget = f"{int(cfg0.get('steps', 10000)):,} steps in {cfg0.get('walltime_min', 7)} min"
    saved: List[Path] = []
    p_seen: set = set()
    n_seen: set = set()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, (camp, label) in enumerate(zip(camps, labels)):
        colour = CATEGORICAL[i % len(CATEGORICAL)]
        cap = read_csv(camp / "capacity.csv")
        xs, ys = [int(r["P"]) for r in cap], [int(r["N_max"]) for r in cap]
        p_seen.update(xs)
        n_seen.update(ys)
        ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=6, label=label)
        for t in read_csv(camp / "thresholds.csv"):
            if "P_fail" in t and "P_last" in t:
                ax.plot([int(t["P_fail"])], [int(t["N"])], linestyle="", marker="o", markersize=10,
                        markerfacecolor="none", markeredgecolor=colour, markeredgewidth=1.5,
                        label=f"{label}: step to P = {t['P_fail']} does not pay off")
                p_seen.add(int(t["P_fail"]))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = sorted(p_seen)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(sorted(n_seen))  # the N grid values that were reached
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("MPI ranks P")
    ax.set_ylabel(f"Largest N that finishes {budget}")
    style(ax)
    legend(ax)
    saved += save(fig, out_dir, f"capacity_{'_vs_'.join(labels)}_largest_N")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    anything = False
    for i, (camp, label) in enumerate(zip(camps, labels)):
        colour = CATEGORICAL[i % len(CATEGORICAL)]
        rows = [r for r in read_csv(camp / "capacity.csv") if r["marginal_in"]]
        rows += [{"P": t["P_fail"], "marginal_in": t["marginal"]} for t in read_csv(camp / "thresholds.csv") if "P_fail" in t]
        rows.sort(key=lambda r: int(r["P"]))
        if rows:
            anything = True
            ax.plot([int(r["P"]) for r in rows], [100 * float(r["marginal_in"]) for r in rows],
                    color=colour, linewidth=2, marker="o", markersize=6, label=label)
    if anything:
        rule = 100 * float(cfg0.get("marginal", 0.8))
        ax.axhline(100, color=GREY, linestyle="--", linewidth=1, label="ideal")
        ax.axhline(rule, color=GREY, linestyle=":", linewidth=1, label=f"rule ({rule:.0f} %)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("MPI ranks P (the step up to P)")
        ax.set_ylabel("Marginal efficiency of the step (%)")
        style(ax)
        legend(ax)
        saved += save(fig, out_dir, f"capacity_{'_vs_'.join(labels)}_marginal_efficiency")
    else:
        plt.close(fig)
    return saved


def make_plots(camps: Sequence[Path], out_dir: Path) -> List[Path]:
    saved: List[Path] = []
    for camp in camps:
        saved += campaign_plots(camp, out_dir)
    saved += isoefficiency_plot(camps, out_dir)
    saved += capacity_plots(camps, out_dir)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plots for isoeff campaigns")
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <campaign>/plots for one campaign, isoeff/plots for several")
    args = parser.parse_args()
    out = args.out or (args.campaigns[0] / "plots" if len(args.campaigns) == 1
                       else Path(__file__).resolve().parent / "plots")
    for path in make_plots(args.campaigns, out):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
