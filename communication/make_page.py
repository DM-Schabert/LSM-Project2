"""Put the strong-scaling measurements of every scheme (C2: one node, C3/C4: two nodes) into communication/results.html.

From the repository root, after copying results.csv off the cluster:

    python3 communication/make_page.py

For every label and N it takes the latest job (one sweep: the serial run and all P) and writes, per P,
the median over the repeats of
    step_us        time per step of the slowest rank (simulation_time_s / iterations)
    step_corr_us   the same, on a node of normal speed (see below)
    forces_us ...  time per step in each timed section, averaged over the ranks (*_time_mean_s)
    forces_max_us  the force kernel of the slowest rank; comm_min_us the communication of the fastest rank

Node speed. The nodes are not equally fast (n-62-21-75 was 12 % faster than the others in every
campaign). The serial runs are the same program on every node, so they measure it: a node's factor is
the median, over N, of its serial step time against the median of all nodes at that N. A run on
several nodes goes at the pace of its slowest node, so step_corr = step / max(factor of its nodes).
Only the data block of the page is replaced; its text and charts compute the rest from it.
"""
import argparse
import datetime
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BLOCK = re.compile(r'(<script id="comm-data" type="application/json">).*?(</script>)', re.S)
CAMPAIGNS = {"C2": ["v1-tile", "v2-ring-tile", "v2-ringblk-tile", "v3-2d-tile"],
             "C3": ["v1-tile-2n", "v2-ring-tile-2n", "v2-ringblk-tile-2n", "v3-2d-tile-2n"]}
RUN_NAME = r"^amdahl_(?P<label>.+?)_j(?P<job>\d+)_r(?P<rep>\d+)$"
PARTS = {"step_us": "simulation_time_s", "forces_us": "forces_time_mean_s", "forces_max_us": "forces_time_s",
         "comm_us": "communication_time_mean_s", "comm_min_us": "communication_time_min_s",
         "sync_us": "sync_time_mean_s", "integrate_us": "integrate_time_mean_s"}


def node_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Relative speed of every node from all serial runs of the Amdahl framework (1.0 = a typical node)."""
    s = df[df["implementation"] == "serial"].copy()
    s["step"] = s["simulation_time_s"] / s["iterations"]
    per = s.groupby(["n_stars_start", "hostname"])["step"].median().rename("step").reset_index()
    per["rel"] = per["step"] / per.groupby("n_stars_start")["step"].transform("median")
    out = per.groupby("hostname").agg(factor=("rel", "median"), sizes=("rel", "size")).reset_index()
    out["runs"] = out["hostname"].map(s.groupby("hostname").size())
    return out.rename(columns={"hostname": "host"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=HERE.parent / "results.csv")
    ap.add_argument("--page", type=Path, default=HERE / "results.html")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df.join(df["run_name"].astype(str).str.extract(RUN_NAME))
    df = df[df["label"].notna()].copy()
    nodes = node_factors(df)
    factor = dict(zip(nodes["host"], nodes["factor"]))
    camp_of = {lab: c for c, labs in CAMPAIGNS.items() for lab in labs}
    df = df[df["label"].isin(camp_of)].copy()
    if df.empty:
        raise SystemExit(f"No rows with the labels {sorted(camp_of)} in {args.results}")
    df["job"] = df["job"].astype(int)
    df["N"] = df["n_stars_start"].astype(int)
    for key, col in PARTS.items():
        df[key] = df[col] / df["iterations"] * 1e6
    hosts = df["hosts"].fillna(df["hostname"]).astype(str)
    df["slowest_factor"] = hosts.map(lambda h: max(factor.get(x, 1.0) for x in h.split(";")))
    df["step_corr_us"] = df["step_us"] / df["slowest_factor"]

    runs, serial = [], []
    for (label, n), d in df.groupby(["label", "N"]):
        d = d[d["job"] == d["job"].max()]                       # latest sweep of this N
        s = d[d["implementation"] == "serial"]
        if len(s):
            serial.append({"label": label, "N": int(n), "job": int(d["job"].iloc[0]), "host": str(s["hostname"].iloc[0]),
                           "step_us": round(float(s["step_us"].median()), 2)})
        for p, g in d[d["implementation"] != "serial"].groupby("n_ranks"):
            row = {"campaign": camp_of[label], "scheme": label.removesuffix("-2n"), "label": label, "N": int(n), "P": int(p),
                   "job": int(g["job"].iloc[0]), "hosts": str(hosts[g.index[0]]), "nodes": int(g["n_nodes"].iloc[0]),
                   "node_factor": round(float(g["slowest_factor"].iloc[0]), 4), "steps": int(g["iterations"].iloc[0]),
                   "repeats": int(len(g)), "decomposition": str(g["decomposition"].iloc[0])}
            row.update({key: round(float(g[key].median()), 2) for key in [*PARTS, "step_corr_us"]})
            runs.append(row)

    data = {"meta": {"made": datetime.date.today().isoformat(), "source": args.results.name, "campaigns": CAMPAIGNS,
                     "cpu": str(df["cpu_model"].dropna().iloc[0]) if df["cpu_model"].notna().any() else ""},
            "runs": runs, "serial": serial,
            "nodes": json.loads(nodes.round(4).to_json(orient="records"))}
    block = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = args.page.read_text()
    page, found = BLOCK.subn(lambda m: m.group(1) + block + m.group(2), page)
    if found != 1:
        raise SystemExit(f"No comm-data block in {args.page}")
    args.page.write_text(page)
    jobs = sorted({r["job"] for r in runs})
    print(f"Wrote {len(runs)} MPI points, {len(serial)} serial runs and {len(nodes)} node factors "
          f"(jobs {jobs[0]}–{jobs[-1]}) into {args.page}")


if __name__ == "__main__":
    main()
