#!/usr/bin/env python3
"""Compare MPI.DOUBLE with a custom XYZ datatype for N-body positions.

Runs independently of the group's simulation. Both variants perform the same
Allgatherv of the same N x 3 C-contiguous float64 array; only the MPI datatype
and its counts/displacements change. No existing repository files are touched.

Usage: mpirun -np 4 python3 benchmark_allgatherv.py --n 5000 \
         --iterations 100 --repeats 5 --output results/bench_manual.csv
"""

import argparse
import csv
import os
from pathlib import Path
import platform
import statistics

import numpy as np
from mpi4py import MPI

from layout import block_partition

FIELDNAMES = [
    "n", "ranks", "method", "repetition", "iterations",
    "time_s", "us_per_call", "full_array_bytes", "host_rank0", "job_id",
]


def make_local_positions(first: int, count: int) -> np.ndarray:
    """Synthetic Gaia-like (N,3) layout. Exact values do not affect MPI bytes."""
    idx = np.arange(first, first + count, dtype=np.float64)
    local = np.empty((count, 3), dtype=np.float64, order="C")
    local[:, 0] = idx + 0.125
    local[:, 1] = -idx - 0.25
    local[:, 2] = idx * 0.5 + 0.375
    return local


def gather_double(comm, local, target, counts, displs):
    # Counts/displacements are in units of float64 (3 doubles per star).
    comm.Allgatherv(
        [local.ravel(), local.size, MPI.DOUBLE],
        [target.ravel(), [3 * c for c in counts],
         [3 * d for d in displs], MPI.DOUBLE],
    )


def gather_xyz(comm, local, target, counts, displs, xyz):
    # One XYZ datatype = three consecutive doubles = one star position.
    # Counts/displacements are now in units of stars, not doubles.
    comm.Allgatherv(
        [local, local.shape[0], xyz],
        [target, counts, displs, xyz],
    )


def benchmark_one(comm, call, iterations: int) -> float:
    """Returns the maximum across ranks of average seconds per call.

    Synchronise before the timed block; do not count Barriers or reductions in
    communication timings. The slowest rank determines collective wall time.
    """
    comm.Barrier()
    start = MPI.Wtime()
    for _ in range(iterations):
        call()
    local_seconds = MPI.Wtime() - start
    max_seconds = comm.allreduce(local_seconds, op=MPI.MAX)
    return max_seconds / iterations


def save_rows(output: str, rows: list[dict]) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="Total stars")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Consecutive Allgatherv calls per repetition")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Repeated timing blocks for each method")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Untimed warmup calls for each method")
    parser.add_argument("--output", default="results/bench_manual.csv",
                        help="CSV output path; only rank 0 writes")
    args = parser.parse_args()
    if args.n < 1 or args.iterations < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("n, iterations, repeats must be >=1 and warmup must be >=0")

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    counts, displs = block_partition(args.n, size)
    local = make_local_positions(displs[rank], counts[rank])
    out_double = np.empty((args.n, 3), dtype=np.float64, order="C")
    out_xyz = np.empty_like(out_double)

    xyz = MPI.DOUBLE.Create_contiguous(3)
    xyz.Commit()
    try:
        # Every position should occupy exactly 3*8=24 bytes; no gaps.
        if xyz.Get_size() != 24 or xyz.Get_extent()[1] != 24:
            raise RuntimeError("Unexpected XYZ datatype size/extent")

        def do_double():
            gather_double(comm, local, out_double, counts, displs)

        def do_xyz():
            gather_xyz(comm, local, out_xyz, counts, displs, xyz)

        # This check catches common displacement and count mistakes before timing.
        do_double()
        do_xyz()
        expected = make_local_positions(0, args.n)
        local_ok = (np.array_equal(out_double, expected)
                    and np.array_equal(out_xyz, expected)
                    and np.array_equal(out_double, out_xyz))
        all_ok = comm.allreduce(local_ok, op=MPI.LAND)
        if not all_ok:
            raise RuntimeError("CORRECTNESS FAILED: exchanged coordinates differ")

        for _ in range(args.warmup):
            do_double()
            do_xyz()

        rows = []
        by_method = {"double": [], "xyz_custom": []}
        calls = {"double": do_double, "xyz_custom": do_xyz}
        for repetition in range(args.repeats):
            # Alternate to reduce the effect of one variant always going first.
            order = ("double", "xyz_custom") if repetition % 2 == 0 else (
                "xyz_custom", "double")
            for method in order:
                sec_per_call = benchmark_one(comm, calls[method], args.iterations)
                by_method[method].append(sec_per_call)
                if rank == 0:
                    rows.append({
                        "n": args.n, "ranks": size, "method": method,
                        "repetition": repetition + 1,
                        "iterations": args.iterations,
                        "time_s": f"{sec_per_call:.12g}",
                        "us_per_call": f"{sec_per_call * 1e6:.6f}",
                        "full_array_bytes": args.n * 3 * np.dtype("float64").itemsize,
                        "host_rank0": platform.node(),
                        "job_id": os.environ.get("LSB_JOBID", "local"),
                    })
        if rank == 0:
            save_rows(args.output, rows)
            std_us = statistics.median(by_method["double"]) * 1e6
            custom_us = statistics.median(by_method["xyz_custom"]) * 1e6
            difference = 100 * (std_us - custom_us) / std_us if std_us else 0.0
            print(f"PASS | N={args.n:,} P={size} | DOUBLE={std_us:.3f} us "
                  f"XYZ={custom_us:.3f} us | XYZ vs DOUBLE={difference:+.2f}% "
                  f"| CSV: {args.output}", flush=True)
    finally:
        xyz.Free()


if __name__ == "__main__":
    main()
