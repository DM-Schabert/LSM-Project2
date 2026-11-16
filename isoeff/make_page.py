"""Put the numbers of one or more isoeff campaigns into isoeff/results.html, a page that opens in any browser.

From the code folder, after the campaigns have finished (or while they run, for a snapshot):

    python3 isoeff/make_page.py isoeff/out/v1-numpy                          # one version
    python3 isoeff/make_page.py isoeff/out/v1-numpy isoeff/out/v2-ring       # compare versions
    python3 isoeff/make_page.py isoeff/out/v* --page /tmp/iso.html           # write somewhere else

Every campaign folder gives config.json, decisions.log and DONE (all written by controller.py), and:
capacity mode: capacity.csv, cap_points.csv and thresholds.csv; time and step mode: points.csv and
thresholds.csv, plus results.csv for the time split per step and the spread over the MPI repeats. Only the data block of the page is replaced; its text
and charts stay as they are. Campaigns appear in the order given, which sets their colours.
The charts load d3 from cdnjs, so the page needs an internet connection.
"""
import argparse
import csv
import datetime
import io
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
BLOCK = re.compile(r'(<script id="isoeff-data" type="application/json">).*?(</script>)', re.S)
RUN_RE = re.compile(r"_N(\d+)_P(\d+)_(serial|mpi)(?:_r\d+)?_j([A-Za-z0-9.]+)$")
CONFIG_KEYS = ("label", "note", "created", "n0", "growth", "max_nodes", "cores_per_node", "marginal",
               "efficiency", "confirm", "at_max_p", "mpi_args", "walltime_min", "mpi_reps", "merge_radius",
               "queue", "map_opts", "steps0", "run_limit_min", "max_walltime_min", "mode", "steps",
               "continued_label", "start_n", "start_p")


def as_csv(rows: List[Dict], fields: List[str]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def num(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def nz(row: Dict[str, str], key: str) -> float:
    """The value, or 0 where it is empty or not a number."""
    x = num(row, key)
    return 0.0 if x != x else x


def breakdown(camp: Path, label: str, min_steps: int) -> Tuple[List[Dict], str]:
    """Per (N, P): median share of the MPI step loop per section, and the spread of S over the repeats."""
    per_job: Dict[str, Dict] = {}
    cpu = ""
    results = camp / "results.csv"
    if not results.exists():
        return [], ""
    with open(results, newline="") as f:
        for row in csv.DictReader(f):
            m = RUN_RE.search(row.get("run_name", ""))
            if not m:
                continue
            try:
                if int(row["iterations"]) < min_steps:
                    continue
            except (KeyError, ValueError):
                continue
            n, p, kind, job = int(m[1]), int(m[2]), m[3], m[4]
            e = per_job.setdefault(job, {"n": n, "p": p, "serial": None, "mpi": []})
            if kind == "serial":
                e["serial"] = num(row, "step_time_mean_ms")
            else:
                e["mpi"].append(row)
                cpu = cpu or row.get("cpu_model", "")
    cells: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for e in per_job.values():
        if e["serial"] is None or not e["mpi"]:
            continue
        c = cells.setdefault((e["n"], e["p"]), {k: [] for k in ("forces", "comm", "sync", "other", "imb", "ms",
                                                                 "s_lo", "s_hi", "reps")})
        times = [num(r, "step_time_mean_ms") for r in e["mpi"]]
        c["s_lo"].append(e["serial"] / max(times))
        c["s_hi"].append(e["serial"] / min(times))
        c["reps"].append(len(times))
        for r in e["mpi"]:
            sim = num(r, "simulation_time_s")
            if not sim > 0:
                continue
            forces = nz(r, "forces_time_s") + nz(r, "integrate_time_s")  # compute on the own rows
            comm = nz(r, "communication_time_s")
            sync = nz(r, "sync_time_s")
            c["forces"].append(forces / sim)
            c["comm"].append(comm / sim)
            c["sync"].append(sync / sim)
            c["other"].append(max(0.0, 1 - (forces + comm + sync) / sim))
            c["imb"].append(num(r, "load_imbalance"))
            c["ms"].append(num(r, "step_time_mean_ms"))
    rows = []
    for (n, p), c in sorted(cells.items()):
        med = lambda k: statistics.median(c[k]) if c[k] else float("nan")  # noqa: E731
        rows.append({"label": label, "N": n, "P": p, "forces": f"{med('forces'):.5f}", "comm": f"{med('comm'):.5f}",
                     "sync": f"{med('sync'):.5f}", "other": f"{med('other'):.5f}", "imbalance": f"{med('imb'):.4f}",
                     "mpi_ms": f"{med('ms'):.4f}", "s_lo": f"{min(c['s_lo']):.4f}", "s_hi": f"{max(c['s_hi']):.4f}",
                     "repeats": sum(c["reps"])})
    return rows, cpu


def mpi_cpu(camp: Path) -> str:
    """The CPU model of the campaign's MPI runs, from results.csv."""
    if (camp / "results.csv").exists():
        with open(camp / "results.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("cpu_model"):
                    return row["cpu_model"]
    return ""


def read_rows(path: Path, label: str) -> Tuple[List[Dict], List[str]]:
    if not path.exists():
        return [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [{"label": label, **r} for r in reader]
        return rows, ["label"] + list(reader.fieldnames or [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaigns", type=Path, nargs="+", help="campaign folders, e.g. isoeff/out/v1-numpy")
    ap.add_argument("--page", type=Path, default=HERE / "results.html", help="the page to fill (default: isoeff/results.html)")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import controller  # for the P list each campaign used

    points, thresholds, splits, campaigns = [], [], [], []
    point_fields, threshold_fields = ["label"], ["label"]
    cap, cap_points, cap_thresholds = [], [], []
    cap_fields, cap_point_fields, cap_threshold_fields = ["label"], ["label"], ["label"]
    labels = set()
    for camp in args.campaigns:
        if not (camp / "config.json").exists():
            raise SystemExit(f"{camp} is not a campaign folder (no config.json)")
        cfg = json.loads((camp / "config.json").read_text())
        label = cfg["label"]
        if label in labels:
            raise SystemExit(f"Campaign '{label}' given twice")
        labels.add(label)
        mode = cfg.get("mode") or ("steps" if cfg.get("steps0") else "time")
        first = "capacity.csv" if mode == "capacity" else "points.csv"
        if not (camp / first).exists():
            raise SystemExit(f"{camp}/{first} not found: run 'controller.py status --campaign {camp}' once")
        if mode == "capacity":  # MPI runs only: capacity per P, every run, the steps that did not pay off
            for name, rows, fields in (("capacity.csv", cap, cap_fields), ("cap_points.csv", cap_points, cap_point_fields),
                                       ("thresholds.csv", cap_thresholds, cap_threshold_fields)):
                r, f = read_rows(camp / name, label)
                rows += r
                fields += [x for x in f if x not in fields]
            cpu = mpi_cpu(camp)
        else:
            p_rows, p_fields = read_rows(camp / "points.csv", label)
            t_rows, t_fields = read_rows(camp / "thresholds.csv", label)
            points += p_rows
            thresholds += t_rows
            point_fields += [f for f in p_fields if f not in point_fields]  # older campaigns lack some columns
            threshold_fields += [f for f in t_fields if f not in threshold_fields]
            split_rows, cpu = breakdown(camp, label, int(cfg.get("min_steps", 3)))
            splits += split_rows
        jobs = json.loads((camp / "jobs.json").read_text()) if (camp / "jobs.json").exists() else []
        done = (camp / "DONE").read_text().strip() if (camp / "DONE").exists() else ""
        log = (camp / "decisions.log").read_text() if (camp / "decisions.log").exists() else ""
        entry = {k: cfg.get(k) for k in CONFIG_KEYS}
        entry["mode"] = mode
        entry.update({"p_list": controller.Campaign(camp).p_list(), "done": done, "log": log,
                      "jobs": len(jobs), "cpu": cpu})
        campaigns.append(entry)

    split_fields = ["label", "N", "P", "forces", "comm", "sync", "other", "imbalance", "mpi_ms", "s_lo", "s_hi", "repeats"]
    data = {"points": as_csv(points, point_fields), "thresholds": as_csv(thresholds, threshold_fields),
            "breakdown": as_csv(splits, split_fields), "campaigns": campaigns,
            "capacity": as_csv(cap, cap_fields), "cap_points": as_csv(cap_points, cap_point_fields),
            "cap_thresholds": as_csv(cap_thresholds, cap_threshold_fields),
            "meta": {"made": datetime.date.today().isoformat()}}
    block = json.dumps(data).replace("</", "<\\/")  # keep "</script>" out of the page's script block

    page = args.page.read_text()
    page, found = BLOCK.subn(lambda m: m.group(1) + block + m.group(2), page)
    if found != 1:
        raise SystemExit(f"No data block in {args.page}")
    args.page.write_text(page)
    print(f"Wrote {len(campaigns)} campaign(s) ({', '.join(c['label'] for c in campaigns)}) into {args.page}")


if __name__ == "__main__":
    main()
