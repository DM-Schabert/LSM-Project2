"""Overlap in depth: does Isend/Irecv hide a transfer behind the force kernel, and when not, why?

Two ranks exchange m bytes in both directions, as two neighbours of the ring do, and compute with
the real force kernel (ParallelImplementation.block_accelerations, unchanged) in small tiles.
For every message size m, five modes:

  comm         Sendrecv alone                                   -> T_comm, the transfer
  compute      the kernel alone, n_tiles tiles                  -> T_comp (about 2 T_comm, at least --min-tiles tiles)
  blocking     Sendrecv, then compute                           (the ring with --ring-comm blocking)
  nonblocking  Irecv + Isend, compute, Waitall                  (the ring with overlap, as Pelle wrote it)
  polling      the same, with Testall after every kernel tile   (a progress call every ~0.1-0.3 ms)

Overlap of a mode = (T_comm + T_comp - T_mode) / T_comm: 1 = the transfer is hidden completely, 0 = not at all.
For the two non-blocking modes the time is also split into posting the requests, computing and waiting.

    mpirun -np 2 --map-by core       --bind-to core python3 communication/overlap/overlap_bench.py   # same node
    mpirun -np 2 --map-by ppr:1:node --bind-to core python3 communication/overlap/overlap_bench.py   # two nodes

Every iteration starts after a barrier; per iteration the slower of the two ranks counts; per
(sweep, size, mode) the median over --reps iterations. The sizes run --sweeps times, every second
sweep in reverse order. One CSV row per (sweep, size, mode).
"""
import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ParallelImplementation as par  # noqa: E402  (the kernel, imported unchanged)

I_START = 10 ** 9   # the i stars are not part of the j block: no self-force masking
TAG = 7
MODES = ["comm", "compute", "blocking", "nonblocking", "polling"]
FIELDS = ["job", "timestamp", "placement", "hosts", "cpus", "mpirun_opts", "sweep", "bytes", "mode", "reps",
          "n_tiles", "tile_us", "total_us", "total_p25_us", "total_p75_us", "post_us", "compute_us", "wait_us", "tests"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "communication" / "overlap" / "out" / "overlap_local.csv")
    ap.add_argument("--placement", default="", help="label (default: intra if both ranks share a host, else inter)")
    ap.add_argument("--min-bytes", type=int, default=64)
    ap.add_argument("--max-bytes", type=int, default=4 << 20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--tile-rows", type=int, default=2, help="kernel tile: rows x --tile-cols pairs (fits the L2 cache)")
    ap.add_argument("--tile-cols", type=int, default=1000)
    ap.add_argument("--compute-factor", type=float, default=2.0, help="T_comp is about this times T_comm")
    ap.add_argument("--min-tiles", type=int, default=4)
    ap.add_argument("--job", default=os.environ.get("LSB_JOBID", "local"))
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != 2:
        raise SystemExit("Run with exactly 2 ranks (mpirun -np 2).")
    other = 1 - rank
    hosts = comm.allgather(MPI.Get_processor_name())
    try:
        cpus = comm.allgather(",".join(map(str, sorted(os.sched_getaffinity(0)))))
    except AttributeError:
        cpus = ["", ""]
    placement = args.placement or ("intra" if hosts[0] == hosts[1] else "inter")

    rng = np.random.default_rng(1)
    j_pos, j_mass = rng.uniform(-50, 50, (args.tile_cols, 3)), rng.uniform(0.5, 2.0, args.tile_cols)
    i_pos = rng.uniform(-50, 50, (args.tile_rows, 3))

    def tile() -> None:
        par.block_accelerations(i_pos, I_START, j_pos, j_mass, 0, 0.01, 0.0, args.tile_rows)

    for _ in range(50):
        tile()
    samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        tile()
        samples.append(time.perf_counter() - t0)
    tile_s = comm.allreduce(float(np.median(samples)), op=MPI.MAX)

    sizes = [2 ** k for k in range(int(math.log2(args.min_bytes)), int(math.log2(args.max_bytes)) + 1)]
    if rank == 0:
        print(f"overlap: {placement} ({' / '.join(hosts)}, cpus {' / '.join(cpus)}), {len(sizes)} sizes "
              f"{args.min_bytes}-{args.max_bytes} B, {args.reps} reps, {args.sweeps} sweeps, kernel tile "
              f"{args.tile_rows} x {args.tile_cols} = {tile_s * 1e6:.0f} us", flush=True)

    rows = []
    for sweep in range(1, args.sweeps + 1):
        for m in (sizes if sweep % 2 else sizes[::-1]):
            send = np.full(m // 8, float(rank))
            recv = np.empty(m // 8)

            def exchange() -> None:
                comm.Sendrecv(send, dest=other, sendtag=TAG, recvbuf=recv, source=other, recvtag=TAG)

            def post():
                return [comm.Irecv(recv, source=other, tag=TAG), comm.Isend(send, dest=other, tag=TAG)]

            for _ in range(5):
                exchange()
            n_tiles = args.min_tiles
            for mode in MODES:
                t_tot, t_post, t_comp, t_wait, tests = [], [], [], [], []
                for _ in range(args.reps + 2):                          # the first two are warm-up
                    comm.Barrier()
                    t0 = time.perf_counter()
                    tp = tc = t0
                    n_tests = 0
                    if mode == "comm":
                        exchange()
                    elif mode == "compute":
                        for _ in range(n_tiles):
                            tile()
                    elif mode == "blocking":
                        exchange()
                        for _ in range(n_tiles):
                            tile()
                    else:
                        reqs = post()
                        tp = time.perf_counter()
                        done = False
                        for _ in range(n_tiles):
                            tile()
                            if mode == "polling" and not done:
                                done = MPI.Request.Testall(reqs)
                                n_tests += 1
                        tc = time.perf_counter()
                        MPI.Request.Waitall(reqs)
                    t1 = time.perf_counter()
                    t_tot.append(t1 - t0)
                    t_post.append(tp - t0)
                    t_comp.append(tc - tp)
                    t_wait.append(t1 - tc if mode in ("nonblocking", "polling") else 0.0)
                    tests.append(n_tests)
                both = comm.gather((t_tot[2:], t_post[2:], t_comp[2:], t_wait[2:], tests[2:]), root=0)
                if rank == 0:
                    a = [np.maximum(np.array(both[0][k]), np.array(both[1][k])) for k in range(5)]
                    q25, q50, q75 = np.percentile(a[0], [25, 50, 75])
                    rows.append({"job": args.job, "timestamp": datetime.now().isoformat(timespec="seconds"),
                                 "placement": placement, "hosts": ";".join(hosts), "cpus": " / ".join(cpus),
                                 "mpirun_opts": os.environ.get("MPIRUN_OPTS", ""), "sweep": sweep, "bytes": m,
                                 "mode": mode, "reps": args.reps, "n_tiles": n_tiles, "tile_us": f"{tile_s * 1e6:.2f}",
                                 "total_us": f"{q50 * 1e6:.3f}", "total_p25_us": f"{q25 * 1e6:.3f}",
                                 "total_p75_us": f"{q75 * 1e6:.3f}", "post_us": f"{np.median(a[1]) * 1e6:.3f}",
                                 "compute_us": f"{np.median(a[2]) * 1e6:.3f}", "wait_us": f"{np.median(a[3]) * 1e6:.3f}",
                                 "tests": f"{np.median(a[4]):.0f}"})
                if mode == "comm":   # compute about compute_factor x the transfer, the same on both ranks
                    t_comm = float(np.median(np.maximum(np.array(both[0][0]), np.array(both[1][0])))) if rank == 0 else 0.0
                    t_comm = comm.bcast(t_comm, root=0)
                    n_tiles = max(args.min_tiles, math.ceil(args.compute_factor * t_comm / tile_s))
            if rank == 0:
                r = {x["mode"]: float(x["total_us"]) for x in rows[-5:]}
                ov = lambda k: (r["comm"] + r["compute"] - r[k]) / r["comm"]  # noqa: E731
                print(f"  sweep {sweep} {m:>8} B: comm {r['comm']:9.1f} us, compute {r['compute']:9.1f} us "
                      f"({n_tiles} tiles); overlap nonblocking {ov('nonblocking'):5.2f}, polling {ov('polling'):5.2f}", flush=True)

    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        new = not args.out.exists() or args.out.stat().st_size == 0
        with args.out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
