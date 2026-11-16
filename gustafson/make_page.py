"""Put the numbers of a Gustafson campaign into gustafson/results.html, a page that opens in any browser.

From the repository root, after the analysis:

    python3 gustafson/gustafson_analysis.py --label v1-numpy --job <the campaign's jobs> --out gustafson/out/v1-numpy
    python3 gustafson/make_page.py --dir gustafson/out/v1-numpy --amdahl amdahl/out/v1-numpy

(run_all.sh's analysis job already writes gustafson/out/<campaign>/; pass that as --dir.)

--dir is the campaign: gustafson_summary.csv, by_N0_<label>.csv, lost_time_summary.csv and
startup_<label>.csv come from there. --amdahl is optional: the output folder of the Amdahl
campaign of the same code version (by_N_<label>.csv and amdahl_summary.csv). With it, the page
compares fixed and grown N for every N that both campaigns ran. Only the data block of the page
is replaced; its text and charts stay as they are. The charts load d3 from cdnjs, so the page
needs an internet connection.
"""
import argparse
import csv
import datetime
import io
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLOCK = re.compile(r'(<script id="gustafson-data" type="application/json">).*?(</script>)', re.S)


def amdahl_loop_rows(summary: Path, by_n: Path) -> str:
    """Only the step-loop rows of the Amdahl campaign's jobs: keeps the page small."""
    with open(by_n, newline="") as f:
        jobs = {r["job"] for r in csv.DictReader(f)}
    with open(summary, newline="") as f:
        reader = csv.DictReader(f)
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator="\n")
        w.writeheader()
        for r in reader:
            if r["measure"] == "loop" and r["job"] in jobs:
                w.writerow(r)
    return out.getvalue()


def main() -> None:
    import sys
    sys.path.insert(0, str(HERE))
    import plan  # the cost model behind the time rule, shown on the page

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="analysis output of the campaign")
    ap.add_argument("--amdahl", type=Path, help="analysis output of the Amdahl campaign (optional)")
    ap.add_argument("--label", default="v1-numpy")
    ap.add_argument("--a-us", type=float, default=plan.A_US_PER_STAR, help="cost model used for the time rule")
    ap.add_argument("--b-ns", type=float, default=plan.B_NS_PER_PAIR)
    ap.add_argument("--page", type=Path, default=HERE / "results.html")
    args = ap.parse_args()

    files = {"summary": args.dir / "gustafson_summary.csv",
             "by_n0": args.dir / f"by_N0_{args.label}.csv",
             "lost": args.dir / "lost_time_summary.csv",
             "startup": args.dir / f"startup_{args.label}.csv"}
    missing = [str(p) for k, p in files.items() if not p.exists() and k != "startup"]
    if missing:
        raise SystemExit("Not found: " + ", ".join(missing) + "\nRun gustafson/gustafson_analysis.py first (see --help).")
    data = {key: path.read_text() for key, path in files.items() if path.exists()}
    if args.amdahl:
        by_n, summary = args.amdahl / f"by_N_{args.label}.csv", args.amdahl / "amdahl_summary.csv"
        if not (by_n.exists() and summary.exists()):
            raise SystemExit(f"Not found: {by_n} or {summary}. Run amdahl/amdahl_analysis.py with --out {args.amdahl}.")
        data["amdahl_by_n"] = by_n.read_text()
        data["amdahl_summary"] = amdahl_loop_rows(summary, by_n)
    data["meta"] = {"label": args.label, "made": datetime.date.today().isoformat(),
                    "a_us": round(args.a_us, 1), "b_ns": round(args.b_ns, 1)}
    block = json.dumps(data).replace("</", "<\\/")      # keep "</script>" out of the page's script block

    page = args.page.read_text()
    page, found = BLOCK.subn(lambda m: m.group(1) + block + m.group(2), page)
    if found != 1:
        raise SystemExit(f"No data block in {args.page}")
    args.page.write_text(page)
    print(f"Wrote the numbers of {args.dir}" + (f" and {args.amdahl}" if args.amdahl else "") + f" into {args.page}")


if __name__ == "__main__":
    main()
