"""Analyse the cache jobs: tables, plots and the verdicts of the expectations written in README.md.

    python3 cache/analyze.py            # the cluster jobs in cache/out/ (files of LOCAL=1 runs are left out)
    python3 cache/analyze.py --local    # only the LOCAL=1 files: checks that the scripts and this analysis run

Inputs (cache/out/): runs_<job>.csv (the program's results rows of K1-K3, copied by the jobs; if there are
none, the cache_ rows of ../results.csv), bandwidth_<job>.csv (K0), comm_load_<job>.csv (K4).
Outputs (cache/out/): k0_bandwidth.csv, k1_tiles.csv (K1 and K2), k3_bign.csv, k4_comm.csv, report.txt and plots/.
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TILE_PAIRS = 1 << 18          # --tile-rows 0 aims at this many pairs per tile (ParallelImplementation.TILE_TARGET_ELEMENTS)
BYTES_PER_PAIR = 79           # temporaries of the kernel per pair of a tile, measured in memory/ (job 30267003)
L2, L3_SHARE = 256e3, 2.5e6   # per core: its L2, and the 30 MB L3 of a socket shared by 12 cores
NAME = re.compile(r"^cache_(K\d)_(allgather|ring|2d)_N(\d+)_P(\d+)_t(\w+)_(close|spread|full|nodes2)_r(\d+)$")


def inputs(pattern: str, local: bool) -> pd.DataFrame:
    fs = sorted(p for p in OUT.glob(pattern) if ("local" in p.name) == local)
    return pd.concat([pd.read_csv(p) for p in fs], ignore_index=True) if fs else pd.DataFrame()


def program_runs(local: bool) -> pd.DataFrame:
    d = inputs("runs_*.csv", local)
    if d.empty and not local and (HERE.parent / "results.csv").exists():
        d = pd.read_csv(HERE.parent / "results.csv", low_memory=False)
        d = d[d["run_name"].astype(str).str.startswith("cache_")]
    if d.empty:
        return d
    m = d["run_name"].astype(str).str.extract(NAME)
    m.columns = ["exp", "scheme", "N", "P", "tile", "place", "rep"]
    d = pd.concat([d.reset_index(drop=True), m.reset_index(drop=True)], axis=1).dropna(subset=["exp"])
    d[["N", "P", "rep"]] = d[["N", "P", "rep"]].astype(int)
    rows = []
    for r in d.itertuples():
        own = math.ceil(r.N / r.P)
        ni, nj, calls = own, r.N, 1
        if r.scheme == "ring":
            nj, calls = own, r.P
        elif r.scheme == "2d":
            g = re.search(r"(\d+)x(\d+)", str(r.decomposition))
            pr, pc = (int(g[1]), int(g[2])) if g else (r.P, 1)
            ni, nj = math.ceil(r.N / pr), math.ceil(r.N / pc)
        k = int(r.kernel_tile_rows) if pd.notna(r.kernel_tile_rows) else 1
        k = min(ni, k if k > 0 else max(1, TILE_PAIRS // nj))
        pairs = ni * r.N if r.scheme == "ring" else ni * nj            # the slowest rank's pairs per step
        it = max(int(r.iterations), 1)
        rows.append({"calls": calls, "tile_rows": k, "tile_cols": nj, "tile_MB": k * nj * BYTES_PER_PAIR / 1e6,
                     "ns_per_pair": r.forces_time_s / it / pairs * 1e9, "step_ms": r.simulation_time_s / it * 1e3,
                     "comm_share": r.communication_time_mean_s / r.simulation_time_s if r.simulation_time_s else np.nan})
    return pd.concat([d, pd.DataFrame(rows)], axis=1)


def medians(d: pd.DataFrame) -> pd.DataFrame:
    keys = ["exp", "scheme", "N", "P", "tile", "place"]
    g = d.groupby(keys, as_index=False)
    out = g.agg(repeats=("rep", "size"), tile_rows=("tile_rows", "first"), tile_cols=("tile_cols", "first"),
                tile_MB=("tile_MB", "first"), calls=("calls", "first"), ns_per_pair=("ns_per_pair", "median"),
                ns_min=("ns_per_pair", "min"), ns_max=("ns_per_pair", "max"), step_ms=("step_ms", "median"),
                comm_share=("comm_share", "median"))
    out["cache"] = np.where(out["tile_MB"] * 1e6 <= L2, "L2", np.where(out["tile_MB"] * 1e6 <= L3_SHARE, "L3 share", "RAM"))
    return out.sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------- the expectations (README.md, written before measuring)

def verdicts(k0: pd.DataFrame, k1: pd.DataFrame, k3: pd.DataFrame, k4: pd.DataFrame) -> list:
    res = []

    def add(eid, text, fn):
        try:
            v, detail = fn()
        except (KeyError, IndexError, ValueError, ZeroDivisionError):
            v, detail = None, "no data"
        res.append((eid, text, "n.a." if v is None else ("yes" if v else "no"), detail))

    def bw(place, p, col="node_GBps_median"):
        r = k0[(k0.test == "triad") & (k0.placement == place) & (k0.n_ranks == p)]
        if r.empty:
            raise KeyError(f"{place} P={p}")
        return float(r[col].median())

    add("E0.1", "One core alone gets 3-15 GB/s (triad, STREAM counting).",
        lambda: (3 <= bw("close", 1, "rank_GBps_median") <= 15, f"{bw('close', 1, 'rank_GBps_median'):.1f} GB/s"))
    add("E0.2", "One socket saturates: 12 cores on it give < 1.2x the bandwidth of 6.",
        lambda: (bw("close", 12) / bw("close", 6) < 1.2, f"{bw('close', 6):.1f} -> {bw('close', 12):.1f} GB/s"))
    add("E0.3", "12 ranks spread over both sockets get >= 1.5x the bandwidth of 12 on one.",
        lambda: (bw("spread", 12) / bw("close", 12) >= 1.5, f"close {bw('close', 12):.1f}, spread {bw('spread', 12):.1f} GB/s"))

    def ns(exp, n, p, tile, place, scheme="allgather"):
        r = k1[(k1.exp == exp) & (k1.scheme == scheme) & (k1.N == n) & (k1.P == p) & (k1.tile == str(tile)) & (k1.place == place)]
        if r.empty:
            raise KeyError(tile)
        return float(r.ns_per_pair.iloc[0])

    def e11():
        v = [ns("K1", 5000, 1, t, "close") for t in (2, 4, 8, 16, 0)]
        return max(v) / min(v) <= 1.15, f"{min(v):.0f}-{max(v):.0f} ns per pair"
    add("E1.1", "On one rank the tile size hardly matters: tiles of 2 rows to 2^18 pairs within 15 % (N = 5 000).", e11)

    def e12():
        out, ok = [], True
        for n in sorted(k1[(k1.exp == "K1") & (k1.P == 24)].N.unique()):
            s = k1[(k1.exp == "K1") & (k1.N == n) & (k1.P == 24) & (k1.tile_MB * 1e6 <= L3_SHARE) & (k1.tile != "1")]
            if s.empty:
                continue
            best, auto = float(s.ns_per_pair.min()), ns("K1", n, 24, 0, "full")
            ok &= best <= 0.87 * auto
            out.append(f"N={n}: {best:.0f} vs {auto:.0f} ns")
        if not out:
            raise KeyError("P=24")
        return ok, "; ".join(out)
    add("E1.2", "On a full node (P = 24) tiles that fit a core's L3 share are >= 13 % faster per pair than 2^18-pair tiles.", e12)

    def e13():
        small = k1[(k1.exp == "K1") & (k1.N == 5000) & (k1.tile_MB * 1e6 <= L3_SHARE) & (k1.tile != "1")]
        best = None
        for t in small.tile.unique():
            try:
                e = (ns("K1", 5000, 1, t, "close")) / ns("K1", 5000, 24, t, "full")   # per-pair time ratio = efficiency
            except KeyError:
                continue
            best = e if best is None else max(best, e)
        e0 = ns("K1", 5000, 1, 0, "close") / ns("K1", 5000, 24, 0, "full")
        return best >= 0.9, f"best small tile {best:.0%}, 2^18-pair tiles {e0:.0%}"
    add("E1.3", "With small tiles the kernel keeps >= 90 % of its one-rank speed per pair on 24 ranks (N = 5 000).", e13)

    def e14():
        out, ok = [], True
        for n in sorted(k1[(k1.exp == "K1") & (k1.P == 24) & (k1.tile == "blk")].N.unique()):
            b, a = ns("K1", n, 24, "blk", "full"), ns("K1", n, 24, 0, "full")
            ok &= b >= a
            out.append(f"N={n}: block {b:.0f} vs {a:.0f} ns")
        if not out:
            raise KeyError("blk")
        return ok, "; ".join(out)
    add("E1.4", "The whole block in one tile is the slowest per pair on a full node.", e14)

    def ratio_k2(n, tiles):
        out = []
        for t in tiles:
            try:
                out.append((t, ns("K1", n, 12, t, "close") / ns("K2", n, 12, t, "spread")))
            except KeyError:
                pass
        if not out:
            raise KeyError(n)
        return out

    def e21():
        r = ratio_k2(5000, [0])
        try:
            r += ratio_k2(20000, [0])
        except KeyError:
            pass
        return all(q >= 1.10 for _, q in r), ", ".join(f"t{t}: close/spread {q:.2f}" for t, q in r)
    add("E2.1", "With 2^18-pair tiles, 12 ranks spread over both sockets are >= 10 % faster per pair than on one socket.", e21)

    def e22():
        r = ratio_k2(5000, [2, 4])
        try:
            r += ratio_k2(20000, [1])
        except KeyError:
            pass
        return all(abs(q - 1) <= 0.05 for _, q in r), ", ".join(f"t{t}: {q:.2f}" for t, q in r)
    add("E2.2", "With small tiles one socket and two are within 5 % of each other.", e22)

    def e31():
        if k3.empty:
            raise KeyError("K3")
        bad = k3[((k3.scheme != "ring") & (k3.comm_share >= 0.02)) | ((k3.scheme == "ring") & (k3.comm_share >= 0.10))]
        return bad.empty, "; ".join(f"{r.scheme} P={r.P} t{r.tile}: {r.comm_share:.1%}" for r in k3.itertuples())
    add("E3.1", "At N = 100 000 communication stays < 2 % of a step (v1, 2D) and < 10 % (ring, mostly waiting).", e31)

    def e32():
        a = k3[(k3.scheme == "allgather") & (k3.P == 24) & (k3.tile == "0")].ns_per_pair.iloc[0]
        b = ns("K1", 20000, 24, 0, "full")
        return abs(a / b - 1) <= 0.15, f"N=100 000: {a:.0f} ns, N=20 000: {b:.0f} ns"
    add("E3.2", "v1's time per pair at N = 100 000 is within 15 % of N = 20 000 (P = 24): positions out of the cache cost little.", e32)

    def e33():
        out, ok = [], True
        for p in sorted(k3.P.unique()):
            s = k3[k3.P == p]
            ring = s[(s.scheme == "ring") & (s.tile != "0")]
            others = s[s.scheme.isin(["allgather", "2d"]) & (s.tile == "0")]
            if ring.empty or others.empty:
                continue
            r, o = float(ring.ns_per_pair.min()), float(others.ns_per_pair.min())
            ok &= r < o
            out.append(f"P={p}: ring small {r:.0f} vs best other {o:.0f} ns")
        if not out:
            raise KeyError("K3")
        return ok, "; ".join(out)
    add("E3.3", "At N = 100 000 the ring with cache-sized pieces is the fastest per pair.", e33)

    def slow(place, load, big=True):
        s = k4[(k4.placement == place)]
        s = s[s.size_bytes >= 65536] if big else s[s.size_bytes <= 1024]
        idle = s[s.load == "idle"].set_index("size_bytes").oneway_median_us
        other = s[s.load == load].set_index("size_bytes").oneway_median_us
        q = (other / idle).dropna()
        if q.empty:
            raise KeyError(load)
        return float(q.median())

    add("E4.1", "On one node, messages >= 64 KB take >= 1.3x as long under the stream and big-tile loads.",
        lambda: (slow("intra", "stream") >= 1.3 and slow("intra", "big") >= 1.3,
                 f"stream x{slow('intra', 'stream'):.2f}, big x{slow('intra', 'big'):.2f}"))
    add("E4.2", "On one node, the small-tile load slows messages >= 64 KB by at most 15 %.",
        lambda: (slow("intra", "small") <= 1.15, f"small x{slow('intra', 'small'):.2f}"))
    add("E4.3", "Across two nodes (RDMA) the stream load slows large messages less than on one node.",
        lambda: (slow("inter", "stream") < slow("intra", "stream"), f"inter x{slow('inter', 'stream'):.2f}, intra x{slow('intra', 'stream'):.2f}"))
    add("E4.4", "Messages <= 1 KB stay within 1.2x of idle under every load (latency, not bandwidth).",
        lambda: (max(slow(p, l, big=False) for p in ("intra", "inter") for l in ("small", "big", "stream")) <= 1.2,
                 f"worst x{max(slow(p, l, big=False) for p in ('intra', 'inter') for l in ('small', 'big', 'stream')):.2f}"))
    return res


# ---------------------------------------------------------------- plots

def plots(k0: pd.DataFrame, k1: pd.DataFrame, k3: pd.DataFrame, k4: pd.DataFrame, where: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    where.mkdir(parents=True, exist_ok=True)
    made = []
    colors = {"1": "#2a78d6", "12": "#eb6834", "24": "#1baf7a", "spread": "#6250d6"}

    if not k0.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for place, c in (("close", "#2a78d6"), ("spread", "#6250d6"), ("full", "#1baf7a")):
            s = k0[(k0.test == "triad") & (k0.placement == place)].sort_values("n_ranks")
            if not s.empty:
                ax.plot(s.n_ranks, s.node_GBps_median, "o-", color=c, label={"close": "one socket first", "spread": "both sockets", "full": "full node"}[place])
        ax.set_xlabel("ranks P"); ax.set_ylabel("memory bandwidth of the node, GB/s (triad)"); ax.set_ylim(bottom=0)
        ax.grid(alpha=.3); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(where / "k0_bandwidth.png", dpi=150); plt.close(fig); made.append("k0_bandwidth.png")

    s1 = k1[(k1.tile != "1")]
    if not s1.empty:
        ns_ = sorted(s1.N.unique())
        fig, axes = plt.subplots(1, len(ns_), figsize=(6 * len(ns_), 4.2), squeeze=False)
        for ax, n in zip(axes[0], ns_):
            for (exp, p, place), g in s1[s1.N == n].groupby(["exp", "P", "place"]):
                g = g.sort_values("tile_MB")
                key = "spread" if place == "spread" else str(p)
                ax.plot(g.tile_MB, g.ns_per_pair, "o--" if place == "spread" else "o-", color=colors.get(key, "#888"),
                        label=f"P = {p}" + (", both sockets" if place == "spread" else ", one socket" if p == 12 else ""))
            for x, lab in ((L2 / 1e6, "L2"), (L3_SHARE / 1e6, "L3 share")):
                ax.axvline(x, color="#9aa1ac", ls=":", lw=1); ax.text(x, ax.get_ylim()[1], f" {lab}", va="top", fontsize=8, color="#5d6573")
            ax.set_xscale("log"); ax.set_xlabel("temporaries of one tile (MB, at 79 B per pair)"); ax.set_ylabel("kernel time per pair (ns), lower is faster")
            ax.set_title(f"N = {n:,}".replace(",", " "), fontsize=10, loc="left"); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(where / "k1_staircase.png", dpi=150); plt.close(fig); made.append("k1_staircase.png")

    if not k3.empty:
        s = k3.sort_values(["P", "ns_per_pair"])
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(s) + 1))
        labels = [f"P={r.P}  {r.scheme}, tiles {r.tile} ({r.tile_MB:.1f} MB)" for r in s.itertuples()]
        ax.barh(range(len(s)), s.ns_per_pair, color=[{"allgather": "#2a78d6", "ring": "#eb6834", "2d": "#1baf7a"}[x] for x in s.scheme])
        for i, r in enumerate(s.itertuples()):
            ax.text(r.ns_per_pair, i, f"  {r.ns_per_pair:.0f} ns, comm {r.comm_share:.1%}", va="center", fontsize=8)
        ax.set_yticks(range(len(s)), labels, fontsize=8); ax.invert_yaxis(); ax.set_xlabel("kernel time per pair (ns), N = 100 000")
        ax.grid(axis="x", alpha=.3); fig.tight_layout(); fig.savefig(where / "k3_bign.png", dpi=150); plt.close(fig); made.append("k3_bign.png")

    if not k4.empty:
        places = [p for p in ("intra", "inter") if p in set(k4.placement)]
        fig, axes = plt.subplots(1, len(places), figsize=(6 * len(places), 4), squeeze=False)
        for ax, place in zip(axes[0], places):
            s = k4[k4.placement == place]
            idle = s[s.load == "idle"].set_index("size_bytes").oneway_median_us
            for load, c in (("small", "#2a78d6"), ("big", "#eb6834"), ("stream", "#6250d6")):
                g = s[s.load == load].set_index("size_bytes").oneway_median_us
                q = (g / idle).dropna()
                if not q.empty:
                    ax.plot(q.index, q.values, "o-", color=c, label=f"load: {load}")
            ax.axhline(1, color="#9aa1ac", lw=1); ax.set_xscale("log")
            ax.set_xlabel("message size (bytes)"); ax.set_ylabel("one-way time / time with idle neighbours")
            ax.set_title("one node, shared memory" if place == "intra" else "two nodes, InfiniBand", fontsize=10, loc="left")
            ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(where / "k4_comm.png", dpi=150); plt.close(fig); made.append("k4_comm.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", action="store_true", help="analyse the LOCAL=1 files only")
    args = ap.parse_args()
    tag = "_local" if args.local else ""
    runs = program_runs(args.local)
    k1 = medians(runs[runs.exp.isin(["K1", "K2"])]) if not runs.empty else pd.DataFrame(columns=["exp", "scheme", "N", "P", "tile", "place", "tile_MB", "ns_per_pair"])
    k3 = medians(runs[runs.exp == "K3"]) if not runs.empty and (runs.exp == "K3").any() else pd.DataFrame(columns=["scheme", "P", "tile", "ns_per_pair", "comm_share", "tile_MB"])
    k0 = inputs("bandwidth_*.csv", args.local)
    k0 = k0.groupby(["placement", "n_ranks", "test"], as_index=False)[["node_GBps_median", "rank_GBps_median"]].median() if not k0.empty else pd.DataFrame(columns=["placement", "n_ranks", "test", "node_GBps_median", "rank_GBps_median"])
    k4 = inputs("comm_load_*.csv", args.local)
    k4 = k4.groupby(["placement", "load", "size_bytes"], as_index=False)[["oneway_median_us", "oneway_q1_us", "oneway_q3_us", "load_rate_median"]].median() if not k4.empty else pd.DataFrame(columns=["placement", "load", "size_bytes", "oneway_median_us"])

    for name, d in (("k0_bandwidth", k0), ("k1_tiles", k1), ("k3_bign", k3), ("k4_comm", k4)):
        if not d.empty:
            d.to_csv(OUT / f"{name}{tag}.csv", index=False)
    made = plots(k0, k1, k3, k4, OUT / f"plots{tag}")
    res = verdicts(k0, k1, k3, k4)
    lines = [f"cache: {len(runs)} program runs, {len(k0)} bandwidth rows, {len(k4)} communication rows" + (" (LOCAL files)" if args.local else ""), ""]
    lines += [f"{eid}  {v:4s}  {text}\n             {detail}" for eid, text, v, detail in res]
    lines += ["", "tables: " + ", ".join(f"out/{n}{tag}.csv" for n in ("k0_bandwidth", "k1_tiles", "k3_bign", "k4_comm")),
              "plots:  " + ", ".join(f"out/plots{tag}/{m}" for m in made)]
    (OUT / f"report{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
