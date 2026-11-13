"""C0: what communication costs on the cluster, without any physics.

Times the communication patterns of our schemes on arrays shaped like ours (float64,
3 values = 24 bytes per star), so that the schemes can be predicted before they are built:

    pingpong    two ranks send a message back and forth: latency and bandwidth (T = mu + s/B)
                for a neighbouring core, the other half of the node and, if there is one, another node
    allreduce   Allreduce of one integer: the cost of a collective that carries no data
                (v1 does this for its stop flag)
    allgatherv  v1: every rank gathers all N positions; blocks split as in ParallelImplementation.py
    ring        P - 1 shifts of a block of ceil(N/P) positions around a periodic 1D Cartesian
                communicator (Sendrecv_replace), i.e. one full turn of the ring scheme
    2d          the traffic of one step of the 2D scheme on a Compute_dims(P, 2) grid: Bcast of a
                block of ceil(N/p_c) positions in the column, Bcast of ceil(N/p_r) positions in the
                row, Allreduce of ceil(N/p_r) forces in the row

Method: every point gets a few untimed warm-up calls, then repeats until --target-s is used
(between --min-reps and --max-reps). A repeat starts after a Barrier and runs as many calls back
to back as last at least 0.1 ms, divided by their number (as the OSU benchmarks do; single calls
of a few microseconds are below the timer's resolution). Every rank times it; a repeat counts
with the slowest rank's time (Allreduce MAX). The CSV keeps the median and the quartiles over the
repeats. Back-to-back calls can overlap a little, which real steps (with computing in between)
do not, so the smallest times are slightly optimistic. With P ranks out of a larger job, the others wait without spinning.
On several nodes, P ranks are spread evenly over the nodes.

Output: one CSV (rank 0 appends a row per point, so a run cut short keeps what it has), by
default communication/results/bench_<LSF job id>.csv. Analyse with communication/bench/analyse_bench.py.

Laptop test (pretends there are two nodes):
    mpirun -np 4 python3 communication/bench/comm_bench.py --fake-nodes 2 --sizes "50 1000" \
        --target-s 0.02 --out /tmp/bench_test.csv
"""
import argparse
import csv
import datetime
import math
import os
import platform
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import mpi4py
import numpy as np
from mpi4py import MPI

STAR_BYTES = 24                     # one position: 3 doubles
BATCH_S = 1e-4                      # a timed repeat lasts at least this long (see time_calls)
REPO = Path(__file__).resolve().parents[2]
COLUMNS = ["job", "started", "n_nodes", "hosts", "pattern", "pair", "n_ranks", "ranks_per_node",
           "n_stars", "message_bytes", "received_bytes_per_rank", "timed", "reps", "calls_per_rep",
           "t_median_us", "t_p25_us", "t_p75_us", "t_min_us", "t_mean_us",
           "mpi_library", "mpi4py", "numpy", "cpu"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", default="50 100 300 1000 3000 10000 30000 100000",
                    help="N for allgatherv, ring and 2d (stars)")
    ap.add_argument("--pingpong-bytes", default=" ".join(str(8 * 2 ** k) for k in range(0, 20)),
                    help="message sizes for pingpong (bytes)")
    ap.add_argument("--p-list", default="2 4 8 12 16 24 32 48",
                    help="rank counts; values above the job size or not divisible by the node count are skipped")
    ap.add_argument("--patterns", default="pingpong allreduce allgatherv ring 2d")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--target-s", type=float, default=0.3, help="timed seconds per point")
    ap.add_argument("--min-reps", type=int, default=20)
    ap.add_argument("--max-reps", type=int, default=2000)
    ap.add_argument("--time-limit", type=float, default=None, help="seconds; no new point is started after it")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fake-nodes", type=int, default=0, help="laptop test only: split the ranks into this many 'nodes'")
    return ap.parse_args()


# ------------------------------------------------------------------ helpers
def quiet_barrier(comm: MPI.Comm) -> None:
    """Barrier in which waiting ranks sleep instead of spinning, so idle ranks leave the cores alone."""
    req = comm.Ibarrier()
    while not req.Test():
        time.sleep(0.0005)


def time_calls(sub: MPI.Comm, op: Callable[[], None], args: argparse.Namespace) -> tuple:
    """Seconds per call for every repeat, each the slowest rank's (the same array on every rank of
    `sub`), and the number of calls per repeat.

    A repeat runs `inner` calls back to back after one Barrier and divides by `inner`, so that it
    lasts at least BATCH_S: single calls of a few microseconds would be lost in the timer's
    resolution and in the ranks leaving the Barrier at slightly different times.
    """
    def run(reps: int, inner: int) -> np.ndarray:
        t = np.empty(reps)
        for k in range(reps):
            sub.Barrier()
            t0 = MPI.Wtime()
            for _ in range(inner):
                op()
            t[k] = (MPI.Wtime() - t0) / inner
        worst = np.empty_like(t)
        sub.Allreduce(t, worst, op=MPI.MAX)
        return worst

    for _ in range(args.warmup):
        op()
    probe = float(np.median(run(5, 1)))            # identical on all ranks, so all pick the same counts
    inner = int(min(1000, max(1, math.ceil(BATCH_S / max(probe, 1e-7)))))
    reps = int(min(args.max_reps, max(args.min_reps, args.target_s / max(probe * inner, 1e-7))))
    return run(reps, inner), inner


def split_blocks(n: int, p: int) -> List[int]:
    """Rows per rank, as numpy.array_split and ParallelImplementation.make_blocks do."""
    base, extra = divmod(n, p)
    return [base + (1 if r < extra else 0) for r in range(p)]


# ------------------------------------------------------------------ patterns (collective over `sub`)
def setup_pingpong(sub: MPI.Comm, nbytes: int):
    buf = np.zeros(nbytes, dtype=np.uint8)
    other = 1 - sub.rank

    def op() -> None:
        if sub.rank == 0:
            sub.Send(buf, dest=other)
            sub.Recv(buf, source=other)
        else:
            sub.Recv(buf, source=other)
            sub.Send(buf, dest=other)
    return op, nbytes, nbytes, 0.5, []         # one-way time = half a round trip


def setup_allreduce(sub: MPI.Comm, n: int):
    a, b = np.ones(1, dtype=np.int64), np.empty(1, dtype=np.int64)
    return (lambda: sub.Allreduce(a, b, op=MPI.MAX)), 8, 8, 1.0, []


def setup_allgatherv(sub: MPI.Comm, n: int):
    rows = split_blocks(n, sub.size)
    counts = [3 * r for r in rows]
    displs = [3 * sum(rows[:r]) for r in range(sub.size)]
    mine = np.full((rows[sub.rank], 3), float(sub.rank))
    out = np.empty((n, 3))
    received = STAR_BYTES * (n - rows[0])
    return (lambda: sub.Allgatherv(mine, [out, counts, displs, MPI.DOUBLE])), STAR_BYTES * n, received, 1.0, []


def setup_ring(sub: MPI.Comm, n: int):
    p = sub.size
    cart = sub.Create_cart([p], periods=[True], reorder=False)
    src, dst = cart.Shift(0, 1)
    b = math.ceil(n / p)
    buf = np.full((b, 3), float(sub.rank))

    def op() -> None:
        for _ in range(p - 1):
            cart.Sendrecv_replace(buf, dest=dst, source=src)
    return op, STAR_BYTES * b, STAR_BYTES * b * (p - 1), 1.0, [cart]


def setup_2d(sub: MPI.Comm, n: int):
    pr, pc = MPI.Compute_dims(sub.size, 2)
    cart = sub.Create_cart([pr, pc], periods=[False, False], reorder=False)
    row, col = cart.Sub([False, True]), cart.Sub([True, False])      # same row / same column
    bi, bj = math.ceil(n / pr), math.ceil(n / pc)
    iblock, jblock = np.ones((bi, 3)), np.ones((bj, 3))
    force, total = np.ones((bi, 3)), np.empty((bi, 3))

    def op() -> None:
        col.Bcast(jblock, root=0)
        row.Bcast(iblock, root=0)
        row.Allreduce(force, total, op=MPI.SUM)
    return op, STAR_BYTES * bi, STAR_BYTES * (2 * bi + bj), 1.0, [row, col, cart]


SETUPS = {"allreduce": setup_allreduce, "allgatherv": setup_allgatherv, "ring": setup_ring, "2d": setup_2d}
TIMED = {"pingpong": "one way (half a round trip)", "allreduce": "one call", "allgatherv": "one call",
         "ring": "one turn (P - 1 shifts)", "2d": "one step (2 Bcast + Allreduce)"}


# ------------------------------------------------------------------ main
def main() -> None:
    args = parse_args()
    world = MPI.COMM_WORLD
    rank, size = world.rank, world.size
    start = MPI.Wtime()

    host = MPI.Get_processor_name()
    if args.fake_nodes > 1:
        host = f"fake{rank * args.fake_nodes // size}"
    hosts = world.allgather(host)
    nodes = list(dict.fromkeys(hosts))
    by_node: Dict[str, List[int]] = {h: [r for r, x in enumerate(hosts) if x == h] for h in nodes}

    job = os.environ.get("LSB_JOBID") or "local" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out = args.out or REPO / "communication" / "results" / f"bench_{job}.csv"
    meta = {"job": job, "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_nodes": len(nodes), "hosts": " ".join(nodes),
            "mpi_library": MPI.Get_library_version().splitlines()[0].strip(),
            "mpi4py": mpi4py.__version__, "numpy": np.__version__, "cpu": cpu_model()}
    if rank == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists() or out.stat().st_size == 0
        fh = open(out, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        print(f"job {job}: {size} ranks on {len(nodes)} node(s) {' '.join(nodes)}; {meta['mpi_library']}", flush=True)
        print(f"writing {out}", flush=True)

    def ranks_for(p: int) -> Optional[List[int]]:
        """p ranks spread evenly over the nodes, the lowest ranks of each node; None if impossible."""
        if p > size or p % len(nodes):
            return None
        k = p // len(nodes)
        if any(len(by_node[h]) < k for h in nodes):
            return None
        return sorted(r for h in nodes for r in by_node[h][:k])

    def measure(pattern: str, ranks: List[int], setup: Callable, value: int, pair: str = "") -> None:
        stop = world.bcast(args.time_limit is not None and MPI.Wtime() - start > args.time_limit
                           if rank == 0 else None, root=0)
        if stop:
            return
        member = rank in ranks
        sub = world.Split(0 if member else MPI.UNDEFINED, key=rank)
        if member:
            op, message, received, scale, extra = setup(sub, value)
            t, inner = time_calls(sub, op, args)
            t = t * scale * 1e6
            for c in extra:
                c.Free()
            sub.Free()
            if rank == 0:
                per_node = {h: sum(1 for r in ranks if hosts[r] == h) for h in nodes}
                row = dict(meta, pattern=pattern, pair=pair, n_ranks=len(ranks),
                           ranks_per_node=max(per_node.values()),
                           n_stars=value if pattern not in ("pingpong", "allreduce") else "",
                           message_bytes=message, received_bytes_per_rank=received, timed=TIMED[pattern],
                           reps=len(t), calls_per_rep=inner, t_median_us=f"{np.median(t):.3f}", t_p25_us=f"{np.percentile(t, 25):.3f}",
                           t_p75_us=f"{np.percentile(t, 75):.3f}", t_min_us=f"{t.min():.3f}", t_mean_us=f"{t.mean():.3f}")
                writer.writerow(row)
                fh.flush()
                what = {"pingpong": f"{pair}, {value:,} B", "allreduce": "1 integer"}.get(pattern, f"N = {value:,}")
                print(f"  {pattern:<10} P = {len(ranks):>3}  {what:<32} {np.median(t):12.2f} us  ({len(t)} x {inner} calls)", flush=True)
        quiet_barrier(world)

    patterns = args.patterns.split()
    sizes = [int(s) for s in args.sizes.split()]
    p_list = [p for p in (int(x) for x in args.p_list.split()) if ranks_for(p) is not None]

    if "pingpong" in patterns:
        first = by_node[nodes[0]]
        pairs = [("neighbouring core", [first[0], first[1]])] if len(first) > 1 else []
        if len(first) >= 4:
            pairs.append(("other half of the node", [first[0], first[len(first) // 2]]))
        if len(nodes) > 1:
            pairs.append(("other node", [first[0], by_node[nodes[1]][0]]))
        for label, pair in pairs:
            for nbytes in (int(b) for b in args.pingpong_bytes.split()):
                measure("pingpong", pair, setup_pingpong, nbytes, pair=label)
    for pattern in [p for p in patterns if p in SETUPS]:
        for p in p_list:
            for n in (sizes if pattern != "allreduce" else [1]):
                if n >= p or pattern == "allreduce":
                    measure(pattern, ranks_for(p), SETUPS[pattern], n)

    if rank == 0:
        fh.close()
        print(f"done after {MPI.Wtime() - start:.0f} s: {out}", flush=True)


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    main()
