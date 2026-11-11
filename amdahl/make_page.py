"""Put the numbers of an Amdahl campaign into amdahl/results.html, a page that opens in any browser.

From the repository root, after the analysis (the page needs both outputs):

    python3 amdahl/amdahl_analysis.py --label v1-numpy --job <one job per N> --out amdahl/out/v1-numpy
    python3 amdahl/amdahl_analysis.py --label v1-numpy --out amdahl/out/v1-numpy-all
    python3 amdahl/make_page.py --dir amdahl/out/v1-numpy --all amdahl/out/v1-numpy-all

--dir is the campaign: by_N_<label>.csv, startup_<label>.csv and lost_time_summary.csv come from
there. --all is the analysis of every job of the label; its amdahl_summary.csv feeds the
reproducibility table (without --all, the summary in --dir is used and the table shows only the
campaign). Only the data block of the page is replaced; its text and charts stay as they are.
The charts load d3 from cdnjs, so the page needs an internet connection.
"""
import argparse
import datetime
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLOCK = re.compile(r'(<script id="amdahl-data" type="application/json">).*?(</script>)', re.S)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="analysis output of the campaign (one job per N)")
    ap.add_argument("--all", type=Path, help="analysis output of every job of the label")
    ap.add_argument("--label", default="v1-numpy")
    ap.add_argument("--page", type=Path, default=HERE / "results.html")
    args = ap.parse_args()

    files = {"summary": (args.all or args.dir) / "amdahl_summary.csv",
             "by_n": args.dir / f"by_N_{args.label}.csv",
             "startup": args.dir / f"startup_{args.label}.csv",
             "lost": args.dir / "lost_time_summary.csv"}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit("Not found: " + ", ".join(missing) + "\nRun amdahl/amdahl_analysis.py first (see --help).")
    data = {key: path.read_text() for key, path in files.items()}
    data["meta"] = {"label": args.label, "made": datetime.date.today().isoformat()}
    block = json.dumps(data).replace("</", "<\\/")      # keep "</script>" out of the page's script block

    page = args.page.read_text()
    page, found = BLOCK.subn(lambda m: m.group(1) + block + m.group(2), page)
    if found != 1:
        raise SystemExit(f"No data block in {args.page}")
    args.page.write_text(page)
    print(f"Wrote the numbers of {args.dir}" + (f" and {args.all}" if args.all else "") + f" into {args.page}")


if __name__ == "__main__":
    main()
