"""Put the overlap measurements into communication/overlap/results.html.

From the repository root, after copying communication/overlap/out/overlap_<job>.csv off the cluster:

    python3 communication/overlap/make_page.py               # the latest job
    python3 communication/overlap/make_page.py --job 30123456

The data block gets (1) per placement, message size and mode the median over the sweeps (and the
fastest and slowest sweep), and (2) the step times of the ring with and without overlap from C2 and
C3 (../../results.csv), so that the page can say how much overlap could ever have saved there.
Only the data block (<script id="overlap-data">) is replaced; the page computes the rest.
"""
import argparse
import datetime
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BLOCK = re.compile(r'(<script id="overlap-data" type="application/json">).*?(</script>)', re.S)
RING = {"v2-ring-tile": ("C2", "overlap"), "v2-ringblk-tile": ("C2", "blocking"),
        "v2-ring-tile-2n": ("C3", "overlap"), "v2-ringblk-tile-2n": ("C3", "blocking")}
TIMES = ["total_us", "post_us", "compute_us", "wait_us"]


def ring_runs(results: Path) -> list:
    """Step time of the ring with and without overlap, per campaign, N and P (latest job, median of the repeats)."""
    if not results.exists():
        return []
    df = pd.read_csv(results)
    df = df.join(df["run_name"].astype(str).str.extract(r"^amdahl_(?P<label>.+?)_j(?P<job>\d+)_r\d+$"))
    # the same node-speed correction as communication/results.html: ring and blocking ring ran on different
    # nodes, and n-62-21-75 is 12 % faster, which would otherwise show up as a +-10 % "effect" of overlap
    spec = importlib.util.spec_from_file_location("comm_page", ROOT / "communication" / "make_page.py")
    comm_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comm_page)
    factor = dict(comm_page.node_factors(df[df["label"].notna()]).set_index("host")["factor"])
    df = df[df["label"].isin(RING) & (df["implementation"] != "serial")].copy()
    df["job"] = df["job"].astype(int)
    slowest = df["hosts"].fillna(df["hostname"]).astype(str).map(lambda h: max(factor.get(x, 1.0) for x in h.split(";")))
    df["step_us"] = df["simulation_time_s"] / df["iterations"] * 1e6 / slowest
    df = df[df["job"] == df.groupby(["label", "n_stars_start"])["job"].transform("max")]
    out = {}
    for (label, n, p), g in df.groupby(["label", "n_stars_start", "n_ranks"]):
        camp, mode = RING[label]
        row = out.setdefault((camp, int(n), int(p)), {"campaign": camp, "N": int(n), "P": int(p), "nodes": int(g["n_nodes"].iloc[0])})
        row[f"{mode}_us"] = round(float(g["step_us"].median()), 2)
    return [r for r in out.values() if "overlap_us" in r and "blocking_us" in r]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", help="the job to show (default: the latest one in communication/overlap/out/)")
    ap.add_argument("--page", type=Path, default=HERE / "results.html")
    args = ap.parse_args()

    files = sorted((HERE / "out").glob("overlap_*.csv"))
    if not files:
        raise SystemExit("No communication/overlap/out/overlap_*.csv yet. Run overlap_batch.sh on the cluster first.")
    df = pd.concat([pd.read_csv(f, dtype={"job": str}) for f in files], ignore_index=True)
    jobs = df.groupby("job")["timestamp"].max().sort_values()
    job = args.job or jobs.index[-1]
    d = df[df["job"] == job]
    if d.empty:
        raise SystemExit(f"No rows of job {job}. Jobs found: {', '.join(jobs.index)}")

    points = []
    for (pl, m, mode), g in d.groupby(["placement", "bytes", "mode"]):
        row = {"placement": pl, "bytes": int(m), "mode": mode, "sweeps": int(g["sweep"].nunique()),
               "n_tiles": int(g["n_tiles"].median()), "tests": float(g["tests"].median())}
        row.update({c: round(float(g[c].median()), 3) for c in TIMES})
        row.update({"total_min_us": round(float(g["total_us"].min()), 3), "total_max_us": round(float(g["total_us"].max()), 3)})
        points.append(row)
    places = d.groupby("placement").agg(hosts=("hosts", "first"), cpus=("cpus", "first"), opts=("mpirun_opts", "first"),
                                        tile_us=("tile_us", "median"), reps=("reps", "first")).reset_index()
    data = {"meta": {"made": datetime.date.today().isoformat(), "job": job, "other_jobs": [j for j in jobs.index if j != job],
                     "placements": json.loads(places.fillna("").to_json(orient="records"))},
            "points": points, "ring": ring_runs(ROOT / "results.csv")}
    block = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = args.page.read_text()
    page, found = BLOCK.subn(lambda mm: mm.group(1) + block + mm.group(2), page)
    if found != 1:
        raise SystemExit(f"No overlap-data block in {args.page}")
    args.page.write_text(page)
    print(f"Wrote {len(points)} points of job {job} ({', '.join(places['placement'])}) and "
          f"{len(data['ring'])} ring runs into {args.page}")


if __name__ == "__main__":
    main()
