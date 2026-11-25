"""C1: compare final states (--save-final CSVs) with a reference run.

For every file: the largest absolute difference of the positions and velocities,
divided by the largest value of the reference (a relative error over the whole
state), and whether the file is bit-identical. Prints one line per file and
appends the same to a CSV, so the result can be copied off the cluster.

    python3 communication/compare_states.py REFERENCE.csv RUN1.csv RUN2.csv ... \
        [--tol 1e-9] [--out communication/out/validate_<job>.csv] [--job JOBID]

Exit code 1 if any file is off by more than --tol or has a different number of stars.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

POS, VEL = ["x_pc", "y_pc", "z_pc"], ["vx_kms", "vy_kms", "vz_kms"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reference", type=Path)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--tol", type=float, default=1e-9, help="largest allowed relative difference (default 1e-9)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--job", default=os.environ.get("LSB_JOBID", "local"))
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    rows, failed = [], False
    for path in args.runs:
        run = pd.read_csv(path)
        same_n = len(run) == len(ref)
        if same_n:
            dp = float(np.max(np.abs(run[POS].to_numpy() - ref[POS].to_numpy())) / np.max(np.abs(ref[POS].to_numpy())))
            dv = float(np.max(np.abs(run[VEL].to_numpy() - ref[VEL].to_numpy())) / np.max(np.abs(ref[VEL].to_numpy())))
            identical = bool(np.array_equal(run[POS + VEL].to_numpy(), ref[POS + VEL].to_numpy()))
        else:
            dp = dv = float("nan")
            identical = False
        ok = same_n and dp <= args.tol and dv <= args.tol
        failed |= not ok
        rows.append({"job": args.job, "reference": args.reference.name, "run": path.name, "n_stars": len(run),
                     "rel_diff_positions": f"{dp:.3e}", "rel_diff_velocities": f"{dv:.3e}",
                     "bit_identical": identical, "tolerance": args.tol, "passed": ok})
        print(f"{'PASSED' if ok else 'FAILED'}  {path.name:<48} positions {dp:9.2e}  velocities {dv:9.2e}"
              f"  {'bit-identical' if identical else ''}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        new = not args.out.exists() or args.out.stat().st_size == 0
        with open(args.out, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
