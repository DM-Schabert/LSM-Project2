"""Put the memory measurements into memory/results.html.

From the repository root, after copying memory/out/memory_<job>.csv and memory/logs/memory_<job>.out off the cluster:

    python3 memory/make_page.py                 # the latest job
    python3 memory/make_page.py --job 30123456

Per launch and phase it writes rank 0, the other ranks (median and largest) and the largest sum over the
ranks of one node. From the job log it takes the node's memory (free -m) and LSF's "Max Memory" of the
job; from ../results.csv the time per star pair of v1 with the node full, so that the page can put the
memory limit next to the time limit. Only the data block (<script id="memory-data">) is replaced.
"""
import argparse
import datetime
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLOCK = re.compile(r'(<script id="memory-data" type="application/json">).*?(</script>)', re.S)
PHASES = ["start", "connected", "loaded", "state", "kernel"]
LAUNCH = ["experiment", "scheme", "N", "P", "tile_rows", "kernel"]


def from_log(job: str) -> dict:
    """Node memory and LSF's Max Memory from memory/logs/memory_<job>.out, if it was copied."""
    path = HERE / "logs" / f"memory_{job}.out"
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    out = {}
    m = re.search(r"^Mem:\s+(\d+)", text, re.M)
    if m:
        out["node_ram_mb"] = int(m.group(1))
    m = re.search(r"Max Memory\s*:\s*([\d.]+)\s*(KB|MB|GB)", text)
    if m:
        out["lsf_max_mb"] = float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[m.group(2)]
    return out


def pair_time() -> float | None:
    """Seconds per star pair and rank with the node full: v1 (tiled kernel), N >= 5 000, P = 24 and 48."""
    path = ROOT / "results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["run_name"].astype(str).str.match(r"^amdahl_v1-tile(-2n)?_j\d+_r\d+$") & (df["implementation"] != "serial")]
    df = df[(df["n_stars_start"] >= 5000) & df["n_ranks"].isin([24, 48])]
    if df.empty:
        return None
    return float((df["simulation_time_s"] / df["iterations"] * df["n_ranks"] / df["n_stars_start"] ** 2).median())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", help="the job to show (default: the latest one in memory/out/)")
    ap.add_argument("--reservation-mb", type=float, default=1024, help="LSF reservation per slot (rusage[mem=1GB])")
    ap.add_argument("--page", type=Path, default=HERE / "results.html")
    args = ap.parse_args()

    files = sorted((HERE / "out").glob("memory_*.csv"))
    if not files:
        raise SystemExit("No memory/out/memory_*.csv yet. Run memory/memory_batch.sh on the cluster first.")
    df = pd.concat([pd.read_csv(f, dtype={"job": str}) for f in files], ignore_index=True)
    jobs = df.groupby("job")["timestamp"].max().sort_values()
    job = args.job or jobs.index[-1]
    d = df[df["job"] == job].copy()
    if d.empty:
        raise SystemExit(f"No rows of job {job}. Jobs found: {', '.join(jobs.index)}")
    # a launch that ran twice in the job: keep the later one
    d["launch_time"] = d.groupby(LAUNCH)["timestamp"].transform("max")
    d = d[d["timestamp"] == d["launch_time"]]

    launches = []
    for key, g in d.groupby(LAUNCH, sort=False):
        row = dict(zip(LAUNCH, [str(key[0]), str(key[1]), int(key[2]), int(key[3]), int(key[4]), str(key[5])]))
        row.update({"nodes": int(g["nodes"].iloc[0]), "kernel_rows": int(g["kernel_rows"].iloc[0]),
                    "kernel_cols": int(g["kernel_cols"].iloc[0]), "seconds": float(g["seconds"].max())})
        for ph in PHASES:
            p = g[g["phase"] == ph]
            r0, rest = p[p["rank"] == 0], p[p["rank"] != 0]
            node_sum = p.groupby("host")[["hwm_mb", "rss_mb"]].sum()
            row[ph] = {"r0_hwm": round(float(r0["hwm_mb"].iloc[0]), 1), "r0_rss": round(float(r0["rss_mb"].iloc[0]), 1),
                       "hwm": round(float(rest["hwm_mb"].median()), 1) if len(rest) else None,
                       "hwm_max": round(float(rest["hwm_mb"].max()), 1) if len(rest) else None,
                       "rss": round(float(rest["rss_mb"].median()), 1) if len(rest) else None,
                       "node_hwm": round(float(node_sum["hwm_mb"].max()), 1), "all_hwm": round(float(p["hwm_mb"].sum()), 1)}
        launches.append(row)

    data = {"meta": {"made": datetime.date.today().isoformat(), "job": job, "hosts": sorted(d["host"].unique().tolist()),
                     "reservation_mb": args.reservation_mb, "pair_s": pair_time(), "other_jobs": [j for j in jobs.index if j != job],
                     **from_log(job)},
            "launches": launches}
    clean = lambda o: ({k: clean(v) for k, v in o.items()} if isinstance(o, dict) else [clean(v) for v in o]  # noqa: E731
                       if isinstance(o, list) else None if isinstance(o, float) and o != o else o)   # NaN -> null
    block = json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = args.page.read_text()
    page, found = BLOCK.subn(lambda mm: mm.group(1) + block + mm.group(2), page)
    if found != 1:
        raise SystemExit(f"No memory-data block in {args.page}")
    args.page.write_text(page)
    print(f"Wrote {len(launches)} launches of job {job} into {args.page}"
          f" (node memory {data['meta'].get('node_ram_mb', '?')} MB, LSF max {data['meta'].get('lsf_max_mb', '?')} MB)")


if __name__ == "__main__":
    main()
