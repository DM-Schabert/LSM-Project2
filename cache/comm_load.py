"""K4: does communication get slower while the other cores load the memory?

Rank 0 and its partner (rank 1 on the same socket, or the first rank of the second node) time a ping-pong
for each message size. Meanwhile every other rank runs one of these loads, until the pair is done:

    idle     sleeps (the reference: nothing else uses the memory)
    small    the program's force kernel, block_accelerations, with small tiles (--small-tile rows, ~1.6 MB)
    big      the same with --tile-rows 0 (about 2^18 pairs, ~20 MB of temporaries per tile)
    stream   a NumPy triad on large arrays: as much memory traffic as NumPy can make

On one node MPI copies a message through shared memory with the CPU, so it competes with the load for the same
memory bandwidth; between nodes the network cards move it (RDMA). One row per (load, size): the one-way time
(half a round trip; median and quartiles) and how much work the load ranks did, as a check that they ran.

    mpirun --map-by ppr:12:package --bind-to core -np 24 python3 cache/comm_load.py --placement intra --partner 1
"""
import argparse
import csv
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ParallelImplementation as par  # noqa: E402  (the kernel is called unchanged)

TAG = 7


def make_load(kind: str, args: argparse.Namespace, rank: int):
    """One unit of the load and its size, so that the rate (pairs/s or GB/s) can be reported."""
    if kind == "idle":
        return (lambda: time.sleep(0.002)), 0.0, ""
    if kind in ("small", "big"):
        rng = np.random.default_rng(rank)
        j = rng.random((args.load_n, 3)) * 100.0                     # positions in pc; values do not change the cost
        i = j[: args.load_rows].copy()
        m = np.ones(args.load_n)
        tile = args.small_tile if kind == "small" else 0
        return (lambda: par.block_accelerations(i, 0, j, m, 0, 0.01, 0.0, tile)), args.load_rows * args.load_n, "Mpairs/s"
    if kind == "stream":
        n = int(args.stream_mb * 1e6 / 8)
        a, b, c = np.zeros(n), np.ones(n), np.full(n, 2.0)
        return (lambda: (np.multiply(c, 1.5, out=a), np.add(a, b, out=a))), 24 * n / 1e3, "GB/s"   # STREAM counting
    raise ValueError(kind)


def pingpong(comm, me: int, partner: int, nbytes: int, seconds: float):
    """One-way times in µs (half of each round trip), timed on rank 0."""
    buf = np.zeros(max(nbytes, 1), dtype=np.uint8)
    first = me == 0

    def round_trip():
        if first:
            comm.Send(buf, dest=partner, tag=TAG)
            comm.Recv(buf, source=partner, tag=TAG)
        else:
            comm.Recv(buf, source=0, tag=TAG)
            comm.Send(buf, dest=0, tag=TAG)

    for _ in range(5):                                                   # warm-up, and a guess of the time
        round_trip()
    if first:
        t0 = time.perf_counter()
        for _ in range(5):
            round_trip()
        reps = int(min(2000, max(20, seconds / max((time.perf_counter() - t0) / 5, 1e-7))))
        comm.send(reps, dest=partner, tag=TAG + 1)
    else:
        for _ in range(5):
            round_trip()
        reps = comm.recv(source=0, tag=TAG + 1)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        round_trip()
        times.append((time.perf_counter() - t0) / 2 * 1e6)
    return np.array(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "cache" / "out" / "comm_load_local.csv")
    ap.add_argument("--job", default=os.environ.get("LSB_JOBID", "local"))
    ap.add_argument("--placement", default="intra", help="label: intra (one node) or inter (two nodes)")
    ap.add_argument("--partner", type=int, default=1, help="rank that answers rank 0 (default 1)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 1024, 5016, 10008, 65536, 524288, 4194304],
                    help="message sizes in bytes; 5016 and 10008 are the ring's blocks at N = 5 000, P = 24 and N = 20 000, P = 48")
    ap.add_argument("--loads", nargs="+", default=["idle", "small", "big", "stream"], choices=["idle", "small", "big", "stream"])
    ap.add_argument("--load-n", type=int, default=5000, help="stars a kernel load computes against (default 5000)")
    ap.add_argument("--load-rows", type=int, default=52, help="own stars per kernel call (default 52, one 2^18-pair tile)")
    ap.add_argument("--small-tile", type=int, default=4, help="rows per tile of the small load (default 4: ~1.6 MB)")
    ap.add_argument("--stream-mb", type=float, default=32, help="MB per array of the stream load (default 32)")
    ap.add_argument("--seconds", type=float, default=0.4, help="time spent on each message size (default 0.4)")
    ap.add_argument("--warmup", type=float, default=0.3, help="seconds the load runs before the pair starts (default 0.3)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if not 0 < args.partner < size:
        raise SystemExit(f"--partner {args.partner} is not a rank of {size}")
    in_pair = rank in (0, args.partner)
    hosts = comm.allgather(socket.gethostname())
    sizes = args.sizes
    rows = []
    for kind in args.loads:
        unit, work, unit_name = make_load(kind, args, rank) if not in_pair else (None, 0.0, "")
        if not in_pair:
            unit()                                                      # first call allocates, outside the timing
        for nbytes in sizes:
            comm.Barrier()
            done = 0
            t_start = time.perf_counter()
            if in_pair:
                time.sleep(args.warmup)                                  # let the load reach its steady state
                times = pingpong(comm, rank, args.partner, nbytes, args.seconds)
                comm.Ibarrier().Wait()
            else:
                stop = comm.Ibarrier()
                while not stop.Test():
                    unit()
                    done += 1
            rate = done * work / (time.perf_counter() - t_start) / 1e6 if work else float("nan")
            rates = comm.gather(rate, root=0)
            if rank == 0:
                load_rates = [r for i, r in enumerate(rates) if i not in (0, args.partner) and np.isfinite(r)]
                rows.append({"job": args.job, "placement": args.placement, "partner": args.partner, "n_ranks": size,
                             "load": kind, "size_bytes": nbytes, "reps": len(times),
                             "oneway_median_us": round(float(np.median(times)), 3),
                             "oneway_q1_us": round(float(np.percentile(times, 25)), 3),
                             "oneway_q3_us": round(float(np.percentile(times, 75)), 3),
                             "bandwidth_MBps": round(nbytes / float(np.median(times)), 2),
                             "load_ranks": size - 2,
                             "load_rate_median": round(float(np.median(load_rates)), 3) if load_rates else "",
                             "load_rate_unit": unit_name if kind != "idle" else "",
                             "host_0": hosts[0], "host_partner": hosts[args.partner]})
        if rank == 0:
            r = [x for x in rows if x["load"] == kind]
            print(f"  {args.placement} {kind:6s}: " + "  ".join(f"{x['size_bytes']} B {x['oneway_median_us']} µs" for x in r), flush=True)
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        new = not args.out.exists()
        with args.out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
