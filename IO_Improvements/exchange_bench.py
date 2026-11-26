"""Time one step's collectives on their own, call by call, without any physics.

Compares the old merge check (one int64 Allreduce on the cluster's
communicator, ParallelImplementation.close_pairs_anywhere) with the step
exchange (one (C + 1, 7) float64 Allreduce on the world communicator,
ClusterImplementation.StepExchanger, buffers allocated once), v1's
allgather_rows with ClusterImplementation.PositionGather (count and offset
lists built once), the bare MPI calls under them, and the cost of a
`with perf.time()` block against timing by hand. Every rank runs the same
calls; the printed time is the slowest rank's mean per call.

On 26 Nov at 24 ranks, the first version of the step exchange (new arrays and
a result object every call) took 27.1 µs per call, the old merge check 13.2 µs,
allgather_rows 31.0 µs and a bare Allreduce of one number 5.3 µs.

Run from the project root, e.g. in a job on a reserved node:
    for P in 24 16 8; do mpirun -np $P python3 IO_Improvements/exchange_bench.py; done
"""

import sys
import time
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
for folder in (ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(folder))

import numpy as np
from mpi4py import MPI

import ClusterImplementation as ci
import ParallelImplementation as pi
from SerialImplementation import Perf, Stars

CALLS: int = 20_000
N_STARS: int = 100

world: MPI.Comm = MPI.COMM_WORLD.Clone()
cluster: MPI.Comm = world.Split(color=0, key=world.Get_rank())
stars: Stars = Stars(positions=np.random.default_rng(1).normal(size=(N_STARS, 3)),
                     velocities=np.zeros((N_STARS, 3)), masses=np.ones(N_STARS),
                     members=np.ones(N_STARS, dtype=np.int64), owner=np.arange(N_STARS),
                     sptypes=np.full(N_STARS, "M"), sptype_guessed=np.zeros(N_STARS, dtype=bool))
state = pi.state_from_stars(stars, cluster)
layout = ci.ClusterLayout(n_clusters=1, ranks_per_cluster=world.Get_size(), cluster_id=0, comm=cluster)
perf: Perf = Perf()


def timeit(label: str, fn) -> None:
    for _ in range(200):  # warm up
        fn()
    world.Barrier()
    start: float = time.perf_counter()
    for _ in range(CALLS):
        fn()
    per_call: float = (time.perf_counter() - start) / CALLS * 1e6
    slowest = world.reduce(per_call, op=MPI.MAX, root=0)
    if world.Get_rank() == 0:
        print(f"  {label:55s} {slowest:7.2f} µs per call")


one: np.ndarray = np.zeros(1, dtype=np.int64)
one_out: np.ndarray = np.empty(1, dtype=np.int64)
table: np.ndarray = np.zeros((2, 7))
table_out: np.ndarray = np.empty((2, 7))
positions: np.ndarray = np.empty((N_STARS, 3))

if world.Get_rank() == 0:
    print(f"===== {world.Get_size()} ranks on {MPI.Get_processor_name()}, {CALLS:,} calls each")
timeit("Allreduce, 1 int64, cluster communicator", lambda: cluster.Allreduce(one, one_out, op=MPI.SUM))
timeit("Allreduce, 1 int64, world communicator", lambda: world.Allreduce(one, one_out, op=MPI.SUM))
timeit("Allreduce, (2, 7) float64, world communicator", lambda: world.Allreduce(table, table_out, op=MPI.SUM))
timeit("Allreduce, (2, 7) float64, cluster communicator", lambda: cluster.Allreduce(table, table_out, op=MPI.SUM))
timeit("old merge check: close_pairs_anywhere", lambda: pi.close_pairs_anywhere(cluster, [], perf))
exchanger = ci.StepExchanger(layout, world, with_summaries=False)
timeit("step exchange, lean: StepExchanger", lambda: exchanger(state, 0, 0, perf))
counts = [int(c) * 3 for c in state.blocks.counts]
displs = [int(d) * 3 for d in state.blocks.displs]
spec = [positions, counts, displs, MPI.DOUBLE]
timeit("bare Allgatherv of 100 positions", lambda: cluster.Allgatherv(state.my_positions, spec))
timeit("Allgatherv via v1's allgather_rows",
       lambda: pi.allgather_rows(cluster, state.my_positions, positions, state.blocks, perf))
gather = ci.PositionGather()
timeit("Allgatherv via PositionGather (lists built once)", lambda: gather(cluster, state, perf))


def timed_block() -> None:
    with perf.time("x"):
        pass


def timed_by_hand() -> None:
    ci.record(perf, "x", time.perf_counter())


timeit("an empty `with perf.time()` block", timed_block)
timeit("timing by hand (perf_counter + record)", timed_by_hand)
cluster.Free()
world.Free()