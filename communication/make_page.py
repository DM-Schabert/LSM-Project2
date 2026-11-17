"""Put the C2 measurements (strong scaling of every scheme on one node) into communication/2d_decomposition.html.

From the repository root, after copying results.csv off the cluster:

    python3 communication/make_page.py
    python3 communication/make_page.py --labels v1-tile v2-ring-tile v2-ringblk-tile v3-2d-tile

For every label and N it takes the latest job (one sweep: the serial run and all P on one node)
and writes, per P, the median over the repeats of
    step_us        time per step of the slowest rank (simulation_time_s / iterations)
    forces_us ...  time per step in each timed section, averaged over the ranks (*_time_mean_s)
    forces_max_us  the same for the slowest rank in the force kernel
plus the serial run of the same job. Only the data block of the page is replaced; its text and
charts stay as they are and compute everything else (speed-up against P = 1, shares) from it.
"""
import argparse
import datetime
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BLOCK = re.compile(r'(<script id="c2-data" type="application/json">).*?(</script>)', re.S)
LABELS = ["v1-tile", "v2-ring-tile", "v2-ringblk-tile", "v3-2d-tile"]
RUN_NAME = r"^amdahl_(?P<label>.+?)_j(?P<job>\d+)_r(?P<rep>\d+)$"
PARTS = {"forces_us": "forces_time_mean_s", "comm_us": "communication_time_mean_s",
         "sync_us": "sync_time_mean_s", "integrate_us": "integrate_time_mean_s",
         "forces_max_us": "forces_time_s", "step_us": "simulation_time_s"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", nargs="+", default=LABELS)
    ap.add_argument("--results", type=Path, default=HERE.parent / "results.csv")
    ap.add_argument("--page", type=Path, default=HERE / "2d_decomposition.html")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df.join(df["run_name"].astype(str).str.extract(RUN_NAME))
    df = df[df["label"].isin(args.labels)].copy()
    if df.empty:
        raise SystemExit(f"No rows with the labels {args.labels} in {args.results}")
    df["job"] = df["job"].astype(int)
    for key, col in PARTS.items():
        df[key] = df[col] / df["iterations"] * 1e6
    df["N"] = df["n_stars_start"].astype(int)

    runs, serial = [], []
    for (label, n), d in df.groupby(["label", "N"]):
        d = d[d["job"] == d["job"].max()]                       # latest sweep of this N
        s = d[d["implementation"] == "serial"]
        if len(s):
            serial.append({"label": label, "N": int(n), "job": int(d["job"].iloc[0]),
                           "host": str(s["hostname"].iloc[0]), "step_us": round(float(s["step_us"].median()), 2)})
        for p, g in d[d["implementation"] != "serial"].groupby("n_ranks"):
            row = {"label": label, "N": int(n), "P": int(p), "job": int(g["job"].iloc[0]),
                   "host": str(g["hostname"].iloc[0]), "steps": int(g["iterations"].iloc[0]), "repeats": int(len(g)),
                   "decomposition": str(g["decomposition"].iloc[0])}
            row.update({key: round(float(g[key].median()), 2) for key in PARTS})
            runs.append(row)

    missing = sorted(set(args.labels) - {r["label"] for r in runs})
    if missing:
        raise SystemExit(f"No MPI runs for {missing} in {args.results}")
    data = {"meta": {"made": datetime.date.today().isoformat(), "source": args.results.name, "labels": args.labels},
            "runs": runs, "serial": serial}
    block = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    page = args.page.read_text()
    page, found = BLOCK.subn(lambda m: m.group(1) + block + m.group(2), page)
    if found != 1:
        raise SystemExit(f"No c2-data block in {args.page}")
    args.page.write_text(page)
    jobs = sorted({r["job"] for r in runs})
    print(f"Wrote {len(runs)} MPI points and {len(serial)} serial runs (jobs {jobs[0]}–{jobs[-1]}) into {args.page}")


if __name__ == "__main__":
    main()
