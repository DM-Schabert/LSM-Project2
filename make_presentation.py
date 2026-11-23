"""Collect the data of every results page into presentation.html.

From the repository root, after the pages themselves are up to date (each folder's make_page.py):

    python3 make_presentation.py

It copies the data blocks of amdahl/, isoeff/, communication/, communication/overlap/ and memory/results.html,
data_types/results/summary.csv and the CSVs of experiments/results/ into the one data block of presentation.html (<script id="deck-data">), keeping only
what the presentation draws. Nothing is measured or fitted again here; the page computes the rest, the same way the
results pages do.
"""
import argparse
import datetime
import io
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DECK = re.compile(r'(<script id="deck-data" type="application/json">).*?(</script>)', re.S)


def block(page: Path, block_id: str) -> dict:
    """The JSON data block of one results page ({} if the page or the block is missing)."""
    if not page.exists():
        return {}
    m = re.search(rf'<script id="{block_id}" type="application/json">(.*?)</script>', page.read_text(), re.S)
    return json.loads(m.group(1)) if m else {}


def csv_rows(text: str) -> list:
    return json.loads(pd.read_csv(io.StringIO(text)).to_json(orient="records")) if text else []


def amdahl() -> dict:
    """v1 on one node: the step loop of the campaign shown on amdahl/results.html (one job per N)."""
    d = block(ROOT / "amdahl" / "results.html", "amdahl-data")
    if not d:
        return {}
    by_n = csv_rows(d.get("by_n", ""))
    jobs = {r["job"] for r in by_n}
    loop = [r for r in csv_rows(d.get("summary", "")) if r["measure"] == "loop" and r["job"] in jobs]
    keep = ["n_stars_start", "steps_requested", "n_ranks", "t_median_s", "speedup", "efficiency"]
    return {"meta": d.get("meta", {}), "by_n": by_n, "loop": [{k: r[k] for k in keep} for r in loop],
            "startup": csv_rows(d.get("startup", ""))}


def isoeff() -> dict:
    d = block(ROOT / "isoeff" / "results.html", "isoeff-data")
    if not d:
        return {}
    camps = [{k: c.get(k) for k in ("label", "mode", "steps", "run_limit_min", "done", "cpu")} for c in d.get("campaigns", [])]
    return {"capacity": csv_rows(d.get("capacity", "")), "campaigns": camps, "meta": d.get("meta", {})}


def merge() -> dict:
    """The merge-balance experiment (experiments/results): stars per rank per step, merge events, timings."""
    res = ROOT / "experiments" / "results"
    read = lambda name: csv_rows((res / name).read_text()) if (res / name).exists() else []  # noqa: E731
    return {"steps": read("diagnostic_steps.csv"), "events": read("merge_events.csv"),
            "timing": read("timing_summary.csv"), "paired": read("paired_comparison.csv")}


def datatypes() -> list:
    path = ROOT / "data_types" / "results" / "summary.csv"
    return csv_rows(path.read_text()) if path.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--page", type=Path, default=ROOT / "presentation.html")
    args = ap.parse_args()
    comm = block(ROOT / "communication" / "results.html", "comm-data")
    data = {"made": datetime.date.today().isoformat(),
            "amdahl": amdahl(), "isoeff": isoeff(),
            "comm": {"meta": comm.get("meta", {}), "runs": comm.get("runs", [])},
            "overlap": block(ROOT / "communication" / "overlap" / "results.html", "overlap-data"),
            "memory": block(ROOT / "memory" / "results.html", "memory-data"),
            "datatypes": datatypes(), "merge": merge()}
    text = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = args.page.read_text()
    page, found = DECK.subn(lambda m: m.group(1) + text + m.group(2), page)
    if found != 1:
        raise SystemExit(f"No deck-data block in {args.page}")
    args.page.write_text(page)
    print(f"Wrote {len(text) // 1024} KB into {args.page}: amdahl {len(data['amdahl'].get('loop', []))} points, "
          f"isoeff {len(data['isoeff'].get('capacity', []))}, communication {len(data['comm']['runs'])} runs, "
          f"overlap {len(data['overlap'].get('points', []))} points, memory {len(data['memory'].get('launches', []))} launches, "
          f"datatypes {len(data['datatypes'])} rows, merge {len(data['merge']['steps'])} steps")


if __name__ == "__main__":
    main()
