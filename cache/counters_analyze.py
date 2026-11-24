"""K5: bytes per pair from RAM (and from L3) for every tile size and shape, from the hardware counters of cache/counters.sh.

    python3 cache/counters_analyze.py            # the cluster job(s) in cache/out/
    python3 cache/counters_analyze.py --local    # only the LOCAL=1 files (times only, no counters on a laptop)

Inputs: cache/out/kbench_<job>.csv (one row per run: label, counter group, pairs) and cache/out/counters_<job>/ (method.txt,
and per run <label>__<group>.csv from likwid or <label>__<group>[_r<rank>].txt from perf). The start-up-only run base_P<P>
of the same group is subtracted, then the counts are divided by the pairs the kernel computed.
Outputs: cache/out/k5_counters.csv, report_k5.txt, plots/k5_bytes.png.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MODEL_READ, MODEL_WRITE = 104, 80          # bytes per pair the NumPy steps of one tile read and write (see README, K5)
LIKWID = re.compile(r"^(Memory|L3|L2) data volume \[GBytes\]( STAT)?,(.*)$")


def likwid_volumes(path: Path) -> dict:
    """{'Memory': bytes, 'L3': bytes, ...} from a likwid-perfctr CSV: the STAT sum if there is one, else the sum of the cores."""
    out, stat = {}, {}
    for line in path.read_text(errors="replace").splitlines():
        m = LIKWID.match(line.strip())
        if not m:
            continue
        nums = [float(x) for x in m[3].split(",") if re.fullmatch(r"-?[0-9.]+(e[-+]?\d+)?", x.strip())]
        if not nums:
            continue
        (stat if m[2] else out)[m[1]] = (nums[0] if m[2] else sum(nums)) * 1e9
    out.update(stat)
    return out


def perf_counts(paths: list) -> dict:
    """{event: value} summed over the files of perf stat -x, (MiB of the memory controllers turned into bytes)."""
    tot = {}
    for p in paths:
        for line in p.read_text(errors="replace").splitlines():
            f = line.split(",")
            if len(f) < 3 or line.startswith("#") or not re.fullmatch(r"[0-9.]+", f[0].strip()):
                continue
            v = float(f[0]) * (2 ** 20 if f[1].strip() == "MiB" else 1)
            ev = "cas_read" if "cas_count_read" in f[2] else "cas_write" if "cas_count_write" in f[2] else f[2].strip()
            tot[ev] = tot.get(ev, 0.0) + v
    return tot


def counts(d: Path, method: str, label: str, tag: str) -> dict:
    """RAM and L3 bytes of one run (None where the method does not give them)."""
    if method == "likwid":
        f = d / f"{label}__{tag}.csv"
        v = likwid_volumes(f) if f.exists() else {}
        return {"ram": v.get("Memory"), "l3": v.get("L3")}
    fs = sorted(d.glob(f"{label}__{tag}.txt")) + sorted(d.glob(f"{label}__{tag}_r*.txt"))
    c = perf_counts(fs) if fs else {}
    if method == "perf-uncore" and "cas_read" in c:
        return {"ram": c["cas_read"] + c.get("cas_write", 0.0), "l3": None, "llc_miss": 64 * (c.get("LLC-load-misses", 0) + c.get("LLC-store-misses", 0))}
    if c:
        return {"ram": 64 * (c.get("LLC-load-misses", 0) + c.get("LLC-store-misses", 0)), "l3": None}
    return {"ram": None, "l3": None}


def analyse(local: bool) -> pd.DataFrame:
    rows = []
    for bench in sorted(OUT.glob("kbench_*.csv")):
        if ("local" in bench.name) != local:
            continue
        job = bench.stem.split("_", 1)[1]
        d = OUT / f"counters_{job}"
        method = (d / "method.txt").read_text().strip() if (d / "method.txt").exists() else "none"
        b = pd.read_csv(bench)
        for r in b[~b.label.str.startswith("base_")].itertuples():
            base = f"base_P{r.ranks}" if f"base_P{r.ranks}" in set(b.label) else None   # the start-up-only run with as many ranks
            c, c0 = counts(d, method, r.label, r.tag), counts(d, method, base, r.tag) if base else {}
            per = lambda k: (c[k] - (c0.get(k) or 0.0)) / r.pairs_counted if c.get(k) is not None and r.pairs_counted else np.nan  # noqa: E731
            rows.append({"job": job, "method": method, "label": r.label, "tag": r.tag, "ranks": r.ranks, "cols": r.cols, "rows": r.rows,
                         "tile_rows": r.tile_rows, "tile_MB": r.tile_MB, "ns_per_pair": r.ns_per_pair_median,
                         "ram_B_per_pair": per("ram"), "l3_B_per_pair": per("l3"), "llc_B_per_pair": per("llc_miss")})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["sweep"] = np.where(t.label.str.startswith("tile_"), "tile", "shape")
    g = t.groupby(["sweep", "label", "ranks", "cols", "tile_rows", "tile_MB"], as_index=False).agg(
        method=("method", "first"), ns_per_pair=("ns_per_pair", "median"), ram_B_per_pair=("ram_B_per_pair", "median"),
        l3_B_per_pair=("l3_B_per_pair", "median"), llc_B_per_pair=("llc_B_per_pair", "median"))
    return g.sort_values(["sweep", "ranks", "tile_MB", "cols"]).reset_index(drop=True)


def verdicts(g: pd.DataFrame) -> list:
    res = []

    def get(label, col):
        r = g[g.label == label]
        if r.empty or pd.isna(r[col].iloc[0]):
            raise KeyError(label)
        return float(r[col].iloc[0])

    def add(eid, text, fn):
        try:
            v, detail = fn()
        except (KeyError, ValueError, ZeroDivisionError):
            v, detail = None, "no data (no counters, or the run is missing)"
        res.append((eid, text, "n.a." if v is None else ("yes" if v else "no"), detail))

    add("E5.1", "With the automatic 20 MB tiles on 24 ranks at least 100 B per pair come from RAM.",
        lambda: (get("tile_P24_c5000_k52", "ram_B_per_pair") >= 100, f"{get('tile_P24_c5000_k52', 'ram_B_per_pair'):.0f} B per pair"))
    add("E5.2", "With 2-row tiles (0.8 MB) on 24 ranks at most 20 B per pair come from RAM.",
        lambda: (get("tile_P24_c5000_k2", "ram_B_per_pair") <= 20, f"{get('tile_P24_c5000_k2', 'ram_B_per_pair'):.0f} B per pair"))
    add("E5.3", "On one rank the 20 MB tile fits the whole 30 MB L3: at most 30 B per pair from RAM.",
        lambda: (get("tile_P1_c5000_k52", "ram_B_per_pair") <= 30, f"{get('tile_P1_c5000_k52', 'ram_B_per_pair'):.0f} B per pair"))

    def e54():
        s = g[(g.sweep == "shape") & (g.ranks == 24) & (g.tile_MB > 10)].dropna(subset=["ns_per_pair"])
        if len(s) < 2:
            raise KeyError("shape")
        ns = s.ns_per_pair.max() / s.ns_per_pair.min()
        ram = s.ram_B_per_pair.dropna()
        if len(ram) < 2:
            return None, f"time per pair varies ×{ns:.2f} with the shape; no RAM counts"
        q = ram.max() / ram.min()
        return q <= 1.2 and ns > 1.2, f"time per pair ×{ns:.2f}, RAM bytes per pair ×{q:.2f} between the shapes"
    add("E5.4", "The shape effect is not RAM traffic: at 20 MB per tile the RAM bytes per pair differ by at most 20 % between shapes, the time by more.", e54)
    return res


def plot(g: pd.DataFrame, where: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    col = {1: "#86b6ef", 12: "#4a8fe0", 24: "#0c3366"}
    tile = g[g.sweep == "tile"]
    for p, s in tile.groupby("ranks"):
        s = s.sort_values("tile_MB")
        if s.ram_B_per_pair.notna().any():
            ax[0].plot(s.tile_MB, s.ram_B_per_pair, "o-", color=col.get(p, "#888"), label=f"{p} rank{'s' if p > 1 else ''}")
    for y, lab in ((MODEL_READ + MODEL_WRITE, "model: every byte to RAM"), (MODEL_READ + 2 * MODEL_WRITE, "… with write-allocate")):
        ax[0].axhline(y, color="#9aa1ac", ls=":", lw=1); ax[0].text(ax[0].get_xlim()[0] if tile.empty else tile.tile_MB.min(), y + 4, lab, fontsize=7, color="#5d6573")
    ax[0].axvline(2.5, color="#9aa1ac", ls="--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_xlabel("tile (MB of temporaries)"); ax[0].set_ylabel("bytes per pair from RAM")
    ax[0].set_title("Where the bytes come from: RAM", fontsize=10, loc="left"); ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=.3)
    shape = g[(g.sweep == "shape") & (g.tile_MB > 10)]
    for p, s in shape.groupby("ranks"):
        s = s.sort_values("cols")
        ax[1].plot(s.cols, s.ns_per_pair, "o-", color=col.get(p, "#888"), label=f"{p} rank{'s' if p > 1 else ''}")
        if s.ram_B_per_pair.notna().any():
            ax[2].plot(s.cols, s.ram_B_per_pair, "o-", color=col.get(p, "#888"), label=f"{p} rank{'s' if p > 1 else ''}")
    for a, t, y in ((ax[1], "Same 20 MB tile, other shape: time", "kernel time per pair (ns)"), (ax[2], "Same 20 MB tile, other shape: RAM bytes", "bytes per pair from RAM")):
        a.set_xscale("log"); a.set_xlabel("columns of a tile (rows × columns ≈ 260 000 pairs)"); a.set_ylabel(y)
        a.set_title(t, fontsize=10, loc="left"); a.grid(alpha=.3); a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    where.mkdir(parents=True, exist_ok=True)
    fig.savefig(where / "k5_bytes.png", dpi=150)
    plt.close(fig)
    return "k5_bytes.png"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    tag = "_local" if args.local else ""
    g = analyse(args.local)
    if g.empty:
        print("No kbench_<job>.csv in cache/out/ yet.")
        return
    g.to_csv(OUT / f"k5_counters{tag}.csv", index=False)
    made = plot(g, OUT / f"plots{tag}")
    lines = [f"K5: {len(g)} settings, counter method: {', '.join(sorted(g.method.unique()))}", "",
             f"model: one tile's NumPy steps read ~{MODEL_READ} B and write ~{MODEL_WRITE} B per pair: ~{MODEL_READ + MODEL_WRITE} B per pair "
             f"if all of it went to RAM, ~{MODEL_READ + 2 * MODEL_WRITE} B with write-allocate", ""]
    for r in g.itertuples():
        ram = f"{r.ram_B_per_pair:7.1f}" if pd.notna(r.ram_B_per_pair) else "      -"
        l3 = f"{r.l3_B_per_pair:7.1f}" if pd.notna(r.l3_B_per_pair) else "      -"
        lines.append(f"  {r.label:24s} {r.ranks:3d} ranks  tile {r.tile_rows:4d} x {r.cols:6d} = {r.tile_MB:7.2f} MB   {r.ns_per_pair:7.1f} ns/pair   RAM {ram} B/pair   L3 {l3} B/pair")
    lines += [""] + [f"{eid}  {v:4s}  {text}\n             {detail}" for eid, text, v, detail in verdicts(g)]
    lines += ["", f"table: out/k5_counters{tag}.csv   plot: out/plots{tag}/{made}"]
    (OUT / f"report_k5{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
