"""Memory in depth: what fills the memory of a rank, and which limit comes first, memory or time?

Runs the real start-up of ParallelImplementation.py (its own functions, unchanged) up to and including
the force kernel, and reads the memory of every rank after each phase from /proc/self/status
(VmHWM: the peak so far, VmRSS: what the rank holds now):

  start      after the imports and MPI_Init (Python, NumPy, pandas, matplotlib, mpi4py, Open MPI)
  connected  after one Alltoall of one number: every pair of ranks has talked once (MPI's memory per peer)
  loaded     after load_and_distribute: rank 0 read the CSV, every rank got the whole catalogue by Bcast
  state      after what main() does next: rank 0 copies the start positions (for --plot); the ring and 2D
             drop the full positions and keep their own blocks; the ring allocates its two travelling buffers
  kernel     after the force kernel: --kernel block = this rank's whole share of one time step, as in a run;
             --kernel tile = one tile only (for large N, where a whole block takes minutes)

    mpirun -np 24 --map-by core --bind-to core python3 memory/memory_probe.py --experiment M1 \\
        --scheme allgather --data gaia_subsets/gaia_nearest_20000.csv --kernel tile

One CSV row per rank and phase. memory/memory_batch.sh runs all experiments in one job.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ParallelImplementation as par  # noqa: E402  (its functions are called unchanged)
import SerialImplementation as serial  # noqa: E402

FIELDS = ["job", "timestamp", "experiment", "scheme", "N", "P", "nodes", "host", "rank", "tile_rows",
          "kernel", "kernel_rows", "kernel_cols", "phase", "hwm_mb", "rss_mb", "seconds"]


def memory_mb() -> tuple:
    """(peak so far, now) in MB of this process; on systems without /proc only the peak (ru_maxrss)."""
    try:
        f = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        return int(f["VmHWM"].split()[0]) / 1024, int(f["VmRSS"].split()[0]) / 1024
    except (OSError, KeyError, ValueError):
        return serial.peak_memory_mb(), float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment", required=True, help="label: M1 (against N), M2 (against tile size), M3 (against P)")
    ap.add_argument("--scheme", choices=("allgather", "ring", "2d"), default="allgather")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tile-rows", type=int, default=0, help="as --tile-rows of ParallelImplementation.py (0 = the default tile)")
    ap.add_argument("--kernel", choices=("block", "tile", "none"), default="tile")
    ap.add_argument("--out", type=Path, default=ROOT / "memory" / "out" / "memory_local.csv")
    ap.add_argument("--job", default=os.environ.get("LSB_JOBID", "local"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    marks = [("start", *memory_mb(), 0.0)]
    world = MPI.COMM_WORLD
    rank, size = world.Get_rank(), world.Get_size()

    def mark(phase: str) -> None:
        marks.append((phase, *memory_mb(), time.perf_counter() - t0))

    world.alltoall([rank] * size)          # every pair has exchanged one message
    mark("connected")

    # the program's own options, exactly as a run with these settings would parse them
    sys.argv = ["ParallelImplementation.py", "--data", args.data, "--decomposition", args.scheme,
                "--merge-radius", "0", "--tile-rows", str(args.tile_rows)]
    prog = par.parse_args()
    comm, ring, grid = world.Dup(), None, None
    if args.scheme == "ring":
        comm = comm.Create_cart([size], periods=[True], reorder=False)
        ring = par.Ring(comm, overlap=True)
    elif args.scheme == "2d":
        comm = comm.Create_cart(par.grid_dims(size, prog.grid_rows), periods=[False, False], reorder=False)
        grid = par.Grid2D(comm)
    perf = serial.Perf() if hasattr(serial, "Perf") else par.Perf()
    state = par.load_and_distribute(prog, comm, perf, grid)
    mark("loaded")

    n = state.count
    full = state.positions
    start_positions = full.copy() if rank == 0 else None            # main() keeps this for --plot
    if grid is not None:                                             # 2D: the column block J_c arrives by Allgatherv
        c0, c1 = grid.cols.range_of(grid.c)
        j_pos, j_mass, j_start = np.array(full[c0:c1]), state.masses[c0:c1], c0
    elif ring is not None:                                           # ring: the travelling block, two buffers
        current, upcoming = ring.buffers(int(state.blocks.counts.max()))
        current[: state.my_count] = state.my_positions
        j_pos, j_mass, j_start = current[: state.my_count], state.masses[state.my_range[0]:state.my_range[1]], state.my_range[0]
    else:                                                            # v1: every position, from the Allgatherv
        j_pos, j_mass, j_start = full, state.masses, 0
    if ring is not None or grid is not None:
        state.positions = full = None                                # as main() does for the ring and 2D
    mark("state")

    i_start, _ = state.my_range
    rows = state.my_count
    if args.kernel == "tile":
        rows = min(rows, par.tile_size(prog.tile_rows, len(j_pos)))
    if args.kernel != "none" and rows > 0:
        par.block_accelerations(state.my_positions[:rows], i_start, j_pos, j_mass, j_start,
                                prog.softening, 0.0, prog.tile_rows)
    mark("kernel")
    del start_positions

    host = MPI.Get_processor_name()
    hosts = world.allgather(host)
    got = world.gather((host, marks), root=0)
    if rank == 0:
        nodes = len(set(hosts))
        out = []
        for r, (h, ms) in enumerate(got):
            for phase, hwm, rss, sec in ms:
                out.append({"job": args.job, "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "experiment": args.experiment, "scheme": args.scheme, "N": n, "P": size, "nodes": nodes,
                            "host": h, "rank": r, "tile_rows": prog.tile_rows, "kernel": args.kernel,
                            "kernel_rows": rows, "kernel_cols": len(j_pos), "phase": phase,
                            "hwm_mb": f"{hwm:.1f}", "rss_mb": f"{rss:.1f}", "seconds": f"{sec:.2f}"})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        new = not args.out.exists() or args.out.stat().st_size == 0
        with args.out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(out)
        others = [ms for r, (_, ms) in enumerate(got) if r != 0] or [got[0][1]]
        peak = lambda ms, ph: next(x[1] for x in ms if x[0] == ph)  # noqa: E731
        print(f"{args.experiment} {args.scheme:>9} N={n:>7} P={size:>2} tile rows {prog.tile_rows:>4} kernel {args.kernel:<5}"
              f" ({rows} x {len(j_pos)}): peak rank 0 {peak(got[0][1], 'kernel'):7.0f} MB, other ranks up to "
              f"{max(peak(ms, 'kernel') for ms in others):7.0f} MB, {time.perf_counter() - t0:5.1f} s", flush=True)


if __name__ == "__main__":
    main()
