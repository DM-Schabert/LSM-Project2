"""K0: how much memory bandwidth the cores of a node get, alone and all at once.

Every rank runs the same NumPy loops on its own arrays (much larger than any cache), all ranks start each
repetition together (Barrier), and rank 0 writes one row per test:

    copy    a = b              counted as 16 B per element (read b, write a)
    triad   a = b + s * c      counted as 24 B per element (read b and c, write a), as STREAM does

NumPy runs the triad as two passes (s * c into a, then a + b), so the hardware moves more bytes than counted:
the numbers are what our NumPy code gets, a lower bound of what the memory system delivers. The node's
bandwidth of one repetition is P * bytes / (time of the slowest rank); the row gives its median over the
repetitions and the median bandwidth of a single rank.

    mpirun --map-by ppr:12:package --bind-to core -np 12 python3 cache/bandwidth.py --placement close
"""
import argparse
import csv
import os
import socket
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent


def package_of_this_rank() -> str:
    """The socket (package) this rank is bound to, from Linux sysfs ('' elsewhere or if not bound to one core)."""
    try:
        cpus = os.sched_getaffinity(0)
        if len(cpus) != 1:
            return ""
        cpu = next(iter(cpus))
        return Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id").read_text().strip()
    except (AttributeError, OSError):
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "cache" / "out" / "bandwidth_local.csv")
    ap.add_argument("--job", default=os.environ.get("LSB_JOBID", "local"))
    ap.add_argument("--placement", default="", help="label written into the row (close, spread, full)")
    ap.add_argument("--mb", type=float, default=64, help="size of each of the three arrays per rank in MB (default 64)")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    n = int(args.mb * 1e6 / 8)
    a, b, c = np.zeros(n), np.ones(n), np.full(n, 2.0)   # written once, so the pages exist before timing
    tests = {"copy": (16, lambda: np.copyto(a, b)),
             "triad": (24, lambda: (np.multiply(c, 1.5, out=a), np.add(a, b, out=a)))}
    where = comm.gather((socket.gethostname(), package_of_this_rank()), root=0)
    rows = []
    for name, (bytes_per_elem, op) in tests.items():
        op()                                              # warm-up
        times = []
        for _ in range(args.reps):
            comm.Barrier()
            t0 = time.perf_counter()
            op()
            times.append(time.perf_counter() - t0)
        all_times = comm.gather(times, root=0)
        if rank == 0:
            per_rep_max = np.max(np.array(all_times), axis=0)            # the slowest rank of each repetition
            node = size * bytes_per_elem * n / per_rep_max / 1e9
            single = bytes_per_elem * n / np.array(all_times).ravel() / 1e9
            rows.append({"job": args.job, "placement": args.placement, "n_ranks": size, "test": name,
                         "mb_per_array": args.mb, "bytes_per_elem": bytes_per_elem, "reps": args.reps,
                         "node_GBps_median": round(float(np.median(node)), 3), "node_GBps_best": round(float(np.max(node)), 3),
                         "rank_GBps_median": round(float(np.median(single)), 3),
                         "hosts": ";".join(sorted({h for h, _ in where})),
                         "packages": ";".join(sorted({p for _, p in where if p})) or "?"})
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        new = not args.out.exists()
        with args.out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print("  ".join(f"{r['test']} P={r['n_ranks']} {r['placement']}: node {r['node_GBps_median']} GB/s, "
                        f"one rank {r['rank_GBps_median']} GB/s, packages {r['packages']}" for r in rows), flush=True)


if __name__ == "__main__":
    main()
