"""K5 (hardware counters): the force kernel alone, on every rank at the same time, for a fixed number of seconds.

Every rank calls ParallelImplementation.block_accelerations, unchanged, on synthetic positions: --rows own stars
against --cols stars, in tiles of --tile-rows rows, over and over for --seconds, and counts the pairs. The ranks start
and stop together. cache/counters.sh reads the hardware counters around the whole program; a run with --seconds 0
does everything except the kernel loop, so the difference of the two is the kernel alone.

Rank 0 appends one row to --out: label, counter group (--tag), ranks, shape, tile size, time per pair and the pairs
of all ranks together (including one warm-up call per rank, which the counters see too).

    mpirun --map-by core --bind-to core -np 24 python3 cache/kernel_bench.py --label tile_k2 --cols 5000 --rows 209 --tile-rows 2
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ParallelImplementation as par  # noqa: E402  (the kernel is called unchanged)

BYTES_PER_PAIR = 79   # temporaries of one tile per pair, measured in memory/


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--tag", default="", help="the counter group of this run (MEM, L3, perf, ...), written into the row")
    ap.add_argument("--cols", type=int, default=5000, help="stars the force comes from (default 5000)")
    ap.add_argument("--rows", type=int, default=209, help="own stars per kernel call (default 209: N = 5000 on 24 ranks)")
    ap.add_argument("--tile-rows", type=int, default=0, help="rows per tile; 0 = automatic (~2^18 pairs), as in the program")
    ap.add_argument("--seconds", type=float, default=12.0, help="length of the kernel loop; 0 = start-up only (the baseline)")
    ap.add_argument("--out", type=Path, default=ROOT / "cache" / "out" / "kbench_local.csv")
    ap.add_argument("--job", default=os.environ.get("LSB_JOBID", "local"))
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rng = np.random.default_rng(rank)
    j = rng.random((args.cols, 3)) * 100.0                     # positions in pc; the values do not change the cost
    i = j[: args.rows].copy() if args.rows <= args.cols else rng.random((args.rows, 3)) * 100.0
    m = np.ones(args.cols)
    call = lambda: par.block_accelerations(i, 0, j, m, 0, 0.01, 0.0, args.tile_rows)  # noqa: E731
    if args.seconds > 0:
        call()                                                 # warm-up: the first call allocates
    comm.Barrier()
    t0 = time.perf_counter()
    calls = 0
    while args.seconds > 0 and time.perf_counter() - t0 < args.seconds:
        call()
        calls += 1
    t = time.perf_counter() - t0
    comm.Barrier()
    per = comm.gather((calls, t), root=0)
    if rank != 0:
        return
    pairs_call = args.rows * args.cols
    k = args.tile_rows if args.tile_rows > 0 else max(1, par.TILE_TARGET_ELEMENTS // args.cols)
    k = min(k, args.rows)
    ns = [tt / c / pairs_call * 1e9 for c, tt in per if c]
    row = {"job": args.job, "label": args.label, "tag": args.tag, "ranks": size, "cols": args.cols, "rows": args.rows,
           "tile_rows": k, "tile_pairs": k * args.cols, "tile_MB": round(k * args.cols * BYTES_PER_PAIR / 1e6, 3),
           "seconds": round(max(tt for _, tt in per), 3), "calls_median": float(np.median([c for c, _ in per])),
           "ns_per_pair_median": round(float(np.median(ns)), 3) if ns else "", "ns_per_pair_max": round(float(max(ns)), 3) if ns else "",
           "pairs_counted": (sum(c for c, _ in per) + (size if args.seconds > 0 else 0)) * pairs_call}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"  {args.label} [{args.tag}] P={size}: {row['ns_per_pair_median']} ns per pair, {row['calls_median']:.0f} calls per rank", flush=True)


if __name__ == "__main__":
    main()
