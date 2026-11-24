"""Cluster decomposition (v3): each group of ranks simulates one cluster of stars.

Cluster_Plan_v3.md, step 2. The input is a CSV with a `cluster` column
(0 .. C-1), e.g. made by IO_Improvements/MakeClusters.py.

How it works
    * The P ranks are split into C equal groups with comm.Split(color=cluster,
      key=rank) [W7]. Ranks 0 .. P/C-1 simulate cluster 0, the next P/C ranks
      cluster 1, and so on. With LSF's span[ptile=24] and P = 24·C, every
      cluster gets exactly one node (the start-up printout shows the hosts).
    * Inside a cluster, the ranks run v1 unchanged: Allgatherv of positions,
      the serial force kernel on their own block, the collective merge path.
      Only the communicator and the stars differ: the cluster's communicator
      and the cluster's stars, with star indices local to the cluster.
    * Between clusters: nothing yet (--far-field off). Every cluster evolves as
      if it were alone, so its result is bit-identical to a serial or v1 run on
      that cluster's stars only; compare_clusters.py checks this. Step 3 adds
      the far field (one Allreduce of cluster summaries per step) where
      far_field_accelerations is now.
    * Stopping (time limit, Ctrl+C, SIGTERM) is agreed on the world
      communicator, so all clusters stop after the same step. Whether to merge
      is decided per cluster, so a cluster without close pairs never waits for
      another cluster's merge.
    * Output (--save-final, --plot, --plot-iter): each cluster is gathered on
      its first rank, then world rank 0 collects the clusters with one
      lower-case gather. This happens once per run (or per live-plot frame);
      step 5 replaces it with MPI-IO.

ParallelImplementation.py is not changed. From it, this file uses only v1's
building blocks (listed in USED_FROM_PARALLEL and checked at import), so the
other decompositions in that file can change freely.

Usage (from the project root; the file finds SerialImplementation.py and
ParallelImplementation.py in the folder above its own):
    python3 IO_Improvements/MakeClusters.py --clusters 2 --n 1000
    mpirun -np 8 python3 IO_Improvements/ClusterImplementation.py \\
        --data gaia_subsets/gaia_clusters_C2_n1000.csv --steps 500 --save-final final_clusters.csv
    mpirun --oversubscribe -np 4 python3 IO_Improvements/ClusterImplementation.py --data ... --quick  # laptop

The number of ranks must be a multiple of the number of clusters.

results.csv gets one row per run, with the same columns as v1 and
implementation=mpi, decomposition=clusters_C<C>, plus n_clusters,
ranks_per_cluster, far_field, cluster_hosts and cluster_forces_max_s (the
slowest rank's force time in each cluster, which shows load imbalance between
clusters, e.g. when only one of them merges). --save-final writes the serial
columns plus `cluster`.
"""

import argparse
import hashlib
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# SerialImplementation.py and ParallelImplementation.py live in the project root,
# one folder above this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpi4py import MPI

import ParallelImplementation as pi
import SerialImplementation as serial
from SerialImplementation import FloatArray, IntArray, Pair, Perf, Stars, Value, format_thousands

# v1's building blocks used here. If ParallelImplementation.py renames one, fail
# at once with a clear message instead of halfway through a job.
USED_FROM_PARALLEL: Tuple[str, ...] = (
    "ParallelState", "state_from_stars", "allgather_rows", "gather_full_state", "close_pairs_anywhere",
    "agree_on_stop", "ParallelStopRequest", "block_accelerations", "cpu_model",
    "STOP_NONE", "STOP_INTERRUPTED", "STOP_WINDOW_CLOSED", "STOP_TIME_LIMIT", "STOP_REASONS")
_missing: List[str] = [name for name in USED_FROM_PARALLEL if not hasattr(pi, name)]
if _missing:
    raise ImportError(f"ParallelImplementation.py has no {', '.join(_missing)}, "
                      f"which ClusterImplementation.py needs")

# Every timed section, in a fixed order, so all ranks reduce the same vector (as in v1)
SECTIONS: Tuple[str, ...] = ("load", "distribute", "communication", "forces", "merge",
                             "integrate", "sync", "plotting", "output")
BYTES_PER_ROW: int = 3 * 8  # one star's x, y, z as float64


# ---------------------------------------------------------------- clusters and communicators

@dataclass
class ClusterLayout:
    """Which cluster this rank works on, and the communicator of that cluster's ranks."""
    n_clusters: int
    ranks_per_cluster: int
    cluster_id: int
    comm: MPI.Comm  # the ranks of this cluster; ranks are in world order

    @property
    def is_leader(self) -> bool:
        """The cluster's first rank: it hands the cluster's state to world rank 0."""
        return self.comm.Get_rank() == 0


def check_cluster_ids(cluster_of: IntArray, n_ranks: int, expected: Optional[int]) -> Optional[str]:
    """Why these cluster labels can't be run on n_ranks ranks, or None if they can."""
    if cluster_of.size == 0:
        return "the file has no stars"
    n_clusters: int = int(cluster_of.max()) + 1
    used: IntArray = np.unique(cluster_of)
    if cluster_of.min() < 0 or len(used) != n_clusters:
        return (f"the cluster column must use every number 0 .. C-1; "
                f"found {', '.join(map(str, used[:10]))}{' ...' if len(used) > 10 else ''}")
    if expected is not None and expected != n_clusters:
        return f"--clusters {expected}, but the file has {n_clusters} clusters"
    if n_ranks % n_clusters != 0:
        return (f"{n_ranks} ranks can't be split evenly over {n_clusters} clusters; "
                f"start a multiple of {n_clusters} ranks")
    return None


def load_clusters(args: argparse.Namespace, world: MPI.Comm, perf: Perf) -> Tuple[Stars, IntArray]:
    """Rank 0 reads the CSV and its cluster column; every rank gets both by Bcast.

    The same start-up as v1 (load_and_distribute), plus the cluster labels.
    Any problem is broadcast, so every rank exits together instead of waiting.
    """
    rank: int = world.Get_rank()
    stars: Optional[Stars] = None
    cluster_of: Optional[IntArray] = None
    error: Optional[str] = None

    with perf.time("load"):
        if rank == 0:
            try:
                if "cluster" not in pd.read_csv(args.data, nrows=0).columns:
                    raise ValueError(f"{args.data} has no 'cluster' column; "
                                     f"make the input with IO_Improvements/MakeClusters.py")
                stars = serial.load_subset(args.data, np.random.default_rng(args.seed))
                cluster_of = pd.read_csv(args.data, usecols=["cluster"])["cluster"].to_numpy(dtype=np.int64)
                error = check_cluster_ids(cluster_of, world.Get_size(), args.clusters)
            except Exception as exc:  # tell the other ranks instead of leaving them waiting
                error = f"{type(exc).__name__}: {exc}"

    with perf.time("distribute"):
        error = world.bcast(error, root=0)
        if error is not None:
            if rank == 0:
                print(f"Could not start: {error}", file=sys.stderr)
            raise SystemExit(1)

        n: int = world.bcast(stars.count if rank == 0 else None, root=0)
        if rank != 0:
            stars = Stars(positions=np.empty((n, 3), dtype=np.float64),
                          velocities=np.empty((n, 3), dtype=np.float64),
                          masses=np.empty(n, dtype=np.float64),
                          members=np.empty(n, dtype=np.int64),
                          owner=np.empty(n, dtype=np.int64),
                          sptypes=np.empty(n, dtype="<U1"),
                          sptype_guessed=np.empty(n, dtype=np.bool_))
            cluster_of = np.empty(n, dtype=np.int64)
        assert stars is not None and cluster_of is not None
        if rank == 0:
            # Bcast sends raw memory: send row-major (C-ordered) arrays only
            stars.positions = np.ascontiguousarray(stars.positions)
            stars.velocities = np.ascontiguousarray(stars.velocities)
        for array in (stars.positions, stars.velocities, stars.masses, stars.members, stars.owner, cluster_of):
            world.Bcast(array, root=0)
        stars.sptypes, stars.sptype_guessed = world.bcast(
            (stars.sptypes, stars.sptype_guessed) if rank == 0 else None, root=0)
        perf.comm_bytes += 0 if rank == 0 else n * (2 * BYTES_PER_ROW + 4 * 8)
    return stars, cluster_of


def stars_of_cluster(stars: Stars, rows: IntArray) -> Stars:
    """The given rows as a Stars object of their own, as if loaded from a file with only these rows.

    Indices are local (0 .. n-1) and owner maps every original star of the
    cluster to itself, exactly what load_subset gives for a one-cluster file.
    """
    n: int = len(rows)
    return Stars(positions=np.ascontiguousarray(stars.positions[rows]),
                 velocities=np.ascontiguousarray(stars.velocities[rows]),
                 masses=stars.masses[rows].copy(),
                 members=stars.members[rows].copy(),
                 owner=np.arange(n, dtype=np.int64),
                 sptypes=stars.sptypes[rows].copy(),
                 sptype_guessed=stars.sptype_guessed[rows].copy())


def cluster_hosts(layout: ClusterLayout, world: MPI.Comm) -> Optional[List[List[str]]]:
    """Collective (once): the host names each cluster runs on, on world rank 0 (None elsewhere)."""
    hosts: List[str] = layout.comm.allgather(MPI.Get_processor_name())
    per_leader: Optional[List[Optional[List[str]]]] = world.gather(
        sorted(set(hosts)) if layout.is_leader else None, root=0)
    if per_leader is None:
        return None
    return [h for h in per_leader if h is not None]  # leaders are in world order = cluster order


# ---------------------------------------------------------------- one step (v1 on the cluster's communicator)

def cluster_accelerations(state: "pi.ParallelState", comm: MPI.Comm, args: argparse.Namespace,
                          perf: Perf) -> Tuple[FloatArray, List[Pair]]:
    """Collective (cluster): forces on the own block from every star of the cluster.

    The same code path as v1's allgather decomposition, on the cluster's
    communicator: with --tile-rows 1 it is bit-identical to the serial code.
    """
    assert state.positions is not None
    with perf.time("communication"):
        pi.allgather_rows(comm, state.my_positions, state.positions, state.blocks, perf)
    start, stop = state.my_range
    with perf.time("forces"):
        if args.tile_rows == 1:
            result: Tuple[FloatArray, List[Pair]] = serial.compute_accelerations(
                state.positions, state.masses, args.softening, args.merge_radius, start, stop)
        else:
            result = pi.block_accelerations(state.positions[start:stop], start, state.positions,
                                            state.masses, 0, args.softening, args.merge_radius,
                                            args.tile_rows)
    perf.interactions += state.my_count * (state.count - 1)  # own rows; summed over all ranks later
    return result


def cluster_merge(state: "pi.ParallelState", my_pairs: List[Pair], comm: MPI.Comm,
                  perf: Perf) -> "pi.ParallelState":
    """Collective (cluster): v1's merge path. Every rank of the cluster merges the same pairs identically."""
    with perf.time("merge"):
        all_pairs: List[Pair] = [pair for rank_pairs in comm.allgather(my_pairs) for pair in rank_pairs]
        perf.comm_bytes += (len(all_pairs) - len(my_pairs)) * 16  # approx.: two int64 per pair
        assert state.positions is not None  # still current: gathered by this step's force pass
        velocities: FloatArray = np.empty((state.count, 3), dtype=np.float64)
        pi.allgather_rows(comm, state.my_velocities, velocities, state.blocks, perf)
        full: Stars = Stars(positions=state.positions, velocities=velocities, masses=state.masses,
                            members=state.members, owner=state.owner,
                            sptypes=state.sptypes, sptype_guessed=state.sptype_guessed)
        merged: Stars = serial.merge_stars(full, all_pairs)
        return pi.state_from_stars(merged, comm)  # new N -> new blocks


def far_field_accelerations(state: "pi.ParallelState", layout: ClusterLayout, world: MPI.Comm,
                            args: argparse.Namespace, perf: Perf) -> Optional[FloatArray]:
    """The pull of the other clusters on the own stars; None while --far-field is off.

    Step 3 (Cluster_Plan_v3.md): one world Allreduce of every cluster's mass,
    centre of mass and rms radius, then G·M_B·(R_B − r_i)/|R_B − r_i|³ for
    every other cluster B.
    """
    return None


def cluster_step(state: "pi.ParallelState", layout: ClusterLayout, world: MPI.Comm,
                 args: argparse.Namespace, perf: Perf) -> Tuple["pi.ParallelState", int]:
    """One time step of this rank's cluster; same order as serial.step and v1.

    Returns the new state and how many stars of this cluster merged away.
    """
    comm: MPI.Comm = layout.comm
    n_before: int = state.count
    accelerations, my_pairs = cluster_accelerations(state, comm, args, perf)

    # Decided per cluster: every rank of a cluster gets the same answer, and the
    # other clusters don't take part, so they never wait for this merge.
    if args.merge_radius > 0:
        while pi.close_pairs_anywhere(comm, my_pairs, perf):
            state = cluster_merge(state, my_pairs, comm, perf)
            accelerations, my_pairs = cluster_accelerations(state, comm, args, perf)

    far: Optional[FloatArray] = far_field_accelerations(state, layout, world, args, perf)

    with perf.time("integrate"):
        if far is not None:
            accelerations += far
        state.my_velocities += accelerations * args.dt
        state.my_positions += state.my_velocities * args.dt
    return state, n_before - state.count


# ---------------------------------------------------------------- output

def gather_world_state(state: "pi.ParallelState", layout: ClusterLayout, world: MPI.Comm,
                       rows_by_cluster: List[IntArray], n_original: int,
                       perf: Perf) -> Tuple[Optional[Stars], Optional[IntArray]]:
    """Collective (world): every cluster's current state, joined in cluster order, on world rank 0.

    Returns (stars, cluster label of every current star) on world rank 0 and
    (None, None) elsewhere. `owner` is rebuilt for the whole file, so --plot
    can still match every original star to the star it ended up in.
    """
    whole_cluster: Stars = pi.gather_full_state(state, layout.comm, perf)  # collective (cluster)
    parts: Optional[List[Optional[Stars]]] = world.gather(whole_cluster if layout.is_leader else None, root=0)
    if parts is None:
        return None, None
    clusters: List[Stars] = [part for part in parts if part is not None]  # world order = cluster order
    assert len(clusters) == layout.n_clusters

    owner: IntArray = np.empty(n_original, dtype=np.int64)
    offset: int = 0
    for rows, part in zip(rows_by_cluster, clusters):
        owner[rows] = part.owner + offset  # cluster-local index -> index in the joined arrays
        offset += part.count
    joined: Stars = Stars(
        positions=np.concatenate([c.positions for c in clusters]),
        velocities=np.concatenate([c.velocities for c in clusters]),
        masses=np.concatenate([c.masses for c in clusters]),
        members=np.concatenate([c.members for c in clusters]),
        owner=owner,
        sptypes=np.concatenate([c.sptypes for c in clusters]),
        sptype_guessed=np.concatenate([c.sptype_guessed for c in clusters]))
    labels: IntArray = np.repeat(np.arange(layout.n_clusters, dtype=np.int64), [c.count for c in clusters])
    return joined, labels


def save_final_state(stars: Stars, cluster_of: IntArray, path: Path) -> None:
    """The serial --save-final columns plus `cluster`, rows in cluster order."""
    pd.DataFrame({
        "x_pc": stars.positions[:, 0], "y_pc": stars.positions[:, 1], "z_pc": stars.positions[:, 2],
        "vx_kms": stars.velocities[:, 0], "vy_kms": stars.velocities[:, 1], "vz_kms": stars.velocities[:, 2],
        "mass_msun": stars.masses,
        "merged_from": stars.members,
        "sptype": stars.sptypes,
        "sptype_guessed": stars.sptype_guessed,
        "cluster": cluster_of,
    }).to_csv(path, index=False)


# ---------------------------------------------------------------- results

def code_fingerprint() -> str:
    """Short hash of the three source files this run used."""
    digest = hashlib.md5()
    for path in (Path(__file__).resolve(), PROJECT_ROOT / "ParallelImplementation.py",
                 PROJECT_ROOT / "SerialImplementation.py"):
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def reduce_cluster_perf(world: MPI.Comm, perf: Perf, layout: ClusterLayout, args: argparse.Namespace,
                        hosts_by_cluster: Optional[List[List[str]]]) -> Tuple[Optional[Perf], Dict[str, Value]]:
    """Collective (world): combine every rank's timings on world rank 0, as v1's reduce_perf does.

    Section and step times are the MAX over ranks (the slowest rank decides when
    a step is done); interactions and bytes are SUMs. Other ranks get (None, {}).
    """
    rank: int = world.Get_rank()
    totals: FloatArray = np.array([perf.totals.get(s, 0.0) for s in SECTIONS], dtype=np.float64)
    t_max, t_min, t_sum = (np.empty_like(totals) for _ in range(3))
    world.Reduce(totals, t_max, op=MPI.MAX, root=0)
    world.Reduce(totals, t_min, op=MPI.MIN, root=0)
    world.Reduce(totals, t_sum, op=MPI.SUM, root=0)

    steps: FloatArray = np.array(perf.step_times, dtype=np.float64)  # same length on every rank
    steps_max: FloatArray = np.empty_like(steps)
    world.Reduce(steps, steps_max, op=MPI.MAX, root=0)

    counters: IntArray = np.array([perf.interactions, perf.comm_bytes], dtype=np.int64)
    counters_sum: IntArray = np.empty_like(counters)
    world.Reduce(counters, counters_sum, op=MPI.SUM, root=0)

    memory: FloatArray = np.array([serial.peak_memory_mb()], dtype=np.float64)
    memory_max: FloatArray = np.empty_like(memory)
    world.Reduce(memory, memory_max, op=MPI.MAX, root=0)

    # The slowest rank's force time in each cluster: shows imbalance between clusters
    forces_by_rank: Optional[List[float]] = world.gather(perf.totals.get("forces", 0.0), root=0)

    if rank != 0:
        return None, {}

    size: int = world.Get_size()
    combined: Perf = Perf()
    combined.totals = defaultdict(float, {s: float(v) for s, v in zip(SECTIONS, t_max)})
    combined.calls = perf.calls
    combined.step_times = steps_max.tolist()
    combined.interactions = int(counters_sum[0])
    combined.comm_bytes = int(counters_sum[1])

    t_mean: FloatArray = t_sum / size
    i_forces: int = SECTIONS.index("forces")
    load_imbalance: float = float(t_max[i_forces] / t_mean[i_forces]) if t_mean[i_forces] > 0 else 1.0
    assert forces_by_rank is not None and hosts_by_cluster is not None
    per_cluster_max: List[float] = [
        max(forces_by_rank[k * layout.ranks_per_cluster:(k + 1) * layout.ranks_per_cluster])
        for k in range(layout.n_clusters)]
    all_hosts: List[str] = sorted({h for hosts in hosts_by_cluster for h in hosts})
    extra: Dict[str, Value] = {
        "load_imbalance": load_imbalance,
        "n_nodes": len(all_hosts),
        "hosts": ";".join(all_hosts),
        "cpu_model": pi.cpu_model(),
        "decomposition": f"clusters_C{layout.n_clusters}",
        "allgather_variant": "v",
        "kernel_tile_rows": args.tile_rows,
        "peak_memory_mb": float(memory_max[0]),
        "lsb_jobid": os.environ.get("LSB_JOBID", ""),
        "code_md5": code_fingerprint(),
        "n_clusters": layout.n_clusters,
        "ranks_per_cluster": layout.ranks_per_cluster,
        "far_field": args.far_field,
        "cluster_hosts": "|".join(f"{k}:{'+'.join(h)}" for k, h in enumerate(hosts_by_cluster)),
        "cluster_forces_max_s": ";".join(f"{t:.3f}" for t in per_cluster_max),
    }
    for s, mn, mean in zip(SECTIONS, t_min, t_mean):
        if s in ("load", "plotting", "output"):  # rank 0 only, min/mean add nothing
            continue
        extra[f"{s}_time_min_s"] = float(mn)
        extra[f"{s}_time_mean_s"] = float(mean)
    extra["sync_time_s"] = float(t_max[SECTIONS.index("sync")])
    extra["distribute_time_s"] = float(t_max[SECTIONS.index("distribute")])
    return combined, extra


# ---------------------------------------------------------------- command line

def non_negative_int(value: str) -> int:
    n: int = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be 0 or larger")
    return n


def parse_args() -> argparse.Namespace:
    """The serial options (same names, same defaults), v1's MPI options, and the cluster ones."""
    parser = argparse.ArgumentParser(
        description="MPI N-body simulation with one cluster of stars per group of ranks (v3). "
                    "Run with mpirun; the number of ranks must be a multiple of the number of clusters.")
    parser.add_argument("--data", type=Path, required=True,
                        help="CSV with a 'cluster' column, e.g. from IO_Improvements/MakeClusters.py")
    parser.add_argument("--quick", action="store_true",
                        help=f"Quick run: stop after at most {serial.QUICK_TIME_LIMIT_S:.0f} s "
                             f"(default without it: {serial.NORMAL_TIME_LIMIT_S / 60:.0f} min)")
    parser.add_argument("--time-limit", type=serial.non_negative_float, default=None, metavar="SECONDS",
                        help="Custom time limit, overrides --quick / the normal limit. "
                             "In batch jobs, keep it below the #BSUB -W walltime")
    parser.add_argument("--steps", type=serial.positive_int, default=5_000,
                        help="Maximum number of time steps (default: 5000)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step in pc/(km/s) ≈ 0.978 Myr (default: 0.01)")
    parser.add_argument("--softening", type=serial.non_negative_float, default=0.01,
                        help="Softening length in pc (default: 0.01)")
    parser.add_argument("--merge-radius", type=serial.non_negative_float, default=0.01,
                        help="Stars closer than this (pc) merge into one; 0 turns merging off (default: 0.01)")
    parser.add_argument("--plot", action="store_true",
                        help="After the run, plot each star's start -> end (world rank 0)")
    parser.add_argument("--plot-iter", type=serial.positive_int, default=None, metavar="N",
                        help="Live plot every N iterations (costs a gather of every cluster each time; "
                             "never use it in benchmarks)")
    parser.add_argument("--plots-dir", type=Path, default=serial.DEFAULT_PLOTS_DIR,
                        help=f"Folder the plots are saved in (default: {serial.DEFAULT_PLOTS_DIR})")
    parser.add_argument("--seed", type=int, default=serial.DEFAULT_SEED,
                        help=f"Seed for the random classes of stars without one (default: {serial.DEFAULT_SEED})")
    parser.add_argument("--save-final", type=Path, default=None,
                        help="Save the final state to this CSV (serial columns plus 'cluster')")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run in the results file "
                             "(default: clusters_<mode>_<data>_<P>ranks)")
    parser.add_argument("--results", type=Path, default=serial.DEFAULT_RESULTS_PATH,
                        help=f"CSV that gets one row per run (default: {serial.DEFAULT_RESULTS_PATH})")
    parser.add_argument("--no-results", action="store_true", help="Don't write to the results file")
    # v1's MPI options
    parser.add_argument("--stop-check-every", type=serial.positive_int, default=1, metavar="K",
                        help="Agree on stopping (time limit / signals) every K steps (default: 1)")
    parser.add_argument("--tile-rows", type=non_negative_int, default=1, metavar="K",
                        help="Force kernel: K own stars per loop iteration (1 = the v1 kernel, "
                             "bit-identical to serial; 0 = automatic). Default: 1")
    # clusters
    parser.add_argument("--clusters", type=serial.positive_int, default=None, metavar="C",
                        help="Expected number of clusters; the run stops if the file has another number "
                             "(default: take it from the file)")
    parser.add_argument("--far-field", choices=("off",), default="off",
                        help="Pull between clusters. off: none, every cluster evolves alone "
                             "(step 3 of Cluster_Plan_v3.md adds the far field). Default: off")
    return parser.parse_args()


# ---------------------------------------------------------------- main

def main() -> None:
    args: argparse.Namespace = parse_args()  # every rank parses the same argv
    world: MPI.Comm = MPI.COMM_WORLD.Clone()  # a private copy of the world communicator [W7]
    rank: int = world.Get_rank()
    size: int = world.Get_size()
    is_root: bool = rank == 0

    run_started: datetime = datetime.now()
    program_start: float = time.perf_counter()
    perf: Perf = Perf()
    mode, time_limit = serial.run_mode(args)
    if args.run_name is None:
        args.run_name = f"clusters_{mode}_{args.data.stem}_{size}ranks"

    stars, cluster_of = load_clusters(args, world, perf)
    n_original: int = stars.count
    n_clusters: int = int(cluster_of.max()) + 1
    rows_by_cluster: List[IntArray] = [np.nonzero(cluster_of == k)[0] for k in range(n_clusters)]

    # One communicator per cluster: consecutive world ranks, in world order [W7]
    ranks_per_cluster: int = size // n_clusters
    cluster_id: int = rank // ranks_per_cluster
    layout: ClusterLayout = ClusterLayout(n_clusters=n_clusters, ranks_per_cluster=ranks_per_cluster,
                                          cluster_id=cluster_id,
                                          comm=world.Split(color=cluster_id, key=rank))
    state: "pi.ParallelState" = pi.state_from_stars(stars_of_cluster(stars, rows_by_cluster[cluster_id]),
                                                    layout.comm)
    hosts_by_cluster: Optional[List[List[str]]] = cluster_hosts(layout, world)  # collective

    start_positions: Optional[FloatArray] = stars.positions.copy() if is_root and args.plot else None
    if is_root:
        assert hosts_by_cluster is not None
        print(f"MPI: {size} ranks = {n_clusters} cluster(s) × {ranks_per_cluster} ranks")
        print(f"Mode: {mode} (time limit {time_limit:.0f} s, at most {format_thousands(args.steps)} steps)")
        print(f"Loaded {format_thousands(n_original)} stars from {args.data}; "
              f"far field: {args.far_field}; kernel tile rows: {args.tile_rows or 'auto'}")
        for k, rows in enumerate(rows_by_cluster):
            print(f"  cluster {k}: {format_thousands(len(rows))} stars, "
                  f"{stars.masses[rows].sum():.1f} M_sun, ranks {k * ranks_per_cluster}-"
                  f"{(k + 1) * ranks_per_cluster - 1}, host(s) {', '.join(hosts_by_cluster[k])}")
        spread: List[int] = [k for k, hosts in enumerate(hosts_by_cluster) if len(hosts) > 1]
        if spread:
            print(f"WARNING: cluster(s) {', '.join(map(str, spread))} run on more than one host. On the HPC, "
                  f"reserve whole nodes (span[ptile=...] with {ranks_per_cluster} ranks per node)")
        if args.merge_radius > 0:
            print(f"Merging stars closer than {args.merge_radius} pc (decided per cluster)")
    del stars  # every rank keeps only its cluster from here on

    live: Optional[serial.LivePlot] = None
    if args.plot_iter is not None:
        with perf.time("plotting"):
            full, _ = gather_world_state(state, layout, world, rows_by_cluster, n_original, perf)
            if is_root:
                assert full is not None
                frames_dir: Path = args.plots_dir / serial.title_to_filename(
                    f"{args.data.stem} clusters live frames")
                live = serial.LivePlot(full, frames_dir)

    deadline: float = program_start + time_limit * (1 - serial.FINISH_RESERVE_FRACTION)
    stop_request = pi.ParallelStopRequest(rank)
    stop_request.install()
    stop_reason: str = "finished"
    steps_done: int = 0

    world.Barrier()  # start the step clock together
    try:
        for i in range(1, args.steps + 1):
            if (i - 1) % args.stop_check_every == 0:
                my_code: int = pi.STOP_INTERRUPTED if stop_request.requested else pi.STOP_NONE
                if is_root:  # only world rank 0's clock and window count: one decision for everyone
                    if live is not None and not live.is_open():
                        my_code = max(my_code, pi.STOP_WINDOW_CLOSED)
                    expected_step_time: float = perf.step_times[-1] if perf.step_times else 0.0
                    if time.perf_counter() + expected_step_time * args.stop_check_every > deadline:
                        my_code = max(my_code, pi.STOP_TIME_LIMIT)
                code: int = pi.agree_on_stop(world, my_code, perf)  # world: all clusters stop together
                if code != pi.STOP_NONE:
                    stop_reason = pi.STOP_REASONS[code]
                    break

            step_start: float = time.perf_counter()
            state, n_merged = cluster_step(state, layout, world, args, perf)
            perf.step_times.append(time.perf_counter() - step_start)
            steps_done = i
            if n_merged > 0 and layout.is_leader:
                print(f"Iteration {format_thousands(i)}: cluster {cluster_id}: {n_merged} star(s) merged, "
                      f"{format_thousands(state.count)} left")

            if args.plot_iter is not None and i % args.plot_iter == 0:
                with perf.time("plotting"):
                    full, _ = gather_world_state(state, layout, world, rows_by_cluster, n_original, perf)
                    if live is not None and live.is_open():
                        assert full is not None
                        live.update(full, iteration=i, sim_time=i * args.dt)
    finally:
        stop_request.restore()

    with perf.time("output"):
        final, final_cluster_of = gather_world_state(state, layout, world, rows_by_cluster, n_original, perf)
    combined, extra = reduce_cluster_perf(world, perf, layout, args, hosts_by_cluster)  # collective
    layout.comm.Free()
    world.Free()  # last collective on our communicators

    if not is_root:
        return

    # ---- world rank 0 only from here: report, save, plot, results row
    assert combined is not None and final is not None and final_cluster_of is not None
    simulation_time: float = float(sum(combined.step_times))  # slowest rank, step by step
    if stop_reason != "finished":
        print(f"Stopped early ({stop_reason}) after {format_thousands(steps_done)} of "
              f"{format_thousands(args.steps)} steps, {time.perf_counter() - program_start:.1f} s "
              f"into a {time_limit:.0f} s limit. Saving what we have.")
    print(f"Stars: {format_thousands(n_original)} at start, {format_thousands(final.count)} at end "
          f"({format_thousands(n_original - final.count)} merged away); per cluster: "
          + ", ".join(f"{k}: {int(np.sum(final_cluster_of == k))}" for k in range(n_clusters)))
    print(f"Final mean distance to centre of mass: {serial.mean_distance_to_centre_of_mass(final):.3f} pc")
    print(f"Total momentum at end: {np.round(serial.total_momentum(final), 10)} M_sun·km/s (starts at 0)")
    print(f"Simulated {format_thousands(steps_done)} steps "
          f"({steps_done * args.dt * serial.MYR_PER_TIME_UNIT:.2f} Myr) in {simulation_time:.2f} s "
          f"({simulation_time / max(steps_done, 1) * 1000:.2f} ms/step, slowest rank)")

    if args.save_final is not None:
        with perf.time("output"):
            save_final_state(final, final_cluster_of, args.save_final)
        print(f"Saved final state to {args.save_final}")

    if live is not None:
        print(f"Saved live plot frames in {live.frames_dir}/")

    if args.plot:
        assert start_positions is not None
        with perf.time("plotting"):
            fig = serial.plot_start_to_end(start_positions, final)
            title: str = (f"{args.data.stem} — clusters {size} ranks — {format_thousands(n_original)} stars — "
                          f"Start to end after {format_thousands(steps_done)} iterations · "
                          f"t = {steps_done * args.dt * serial.MYR_PER_TIME_UNIT:.2f} Myr")
            if final.count < n_original:
                title += f" — {format_thousands(final.count)} after merging"
            saved_plot: Path = serial.save_figure(fig, args.plots_dir, title)
        print(f"Saved plot: {saved_plot}")

    total_time: float = time.perf_counter() - program_start
    combined.totals["output"] = max(combined.totals["output"], perf.totals["output"])
    combined.totals["plotting"] = max(combined.totals["plotting"], perf.totals["plotting"])

    row: Dict[str, Value] = serial.build_results_row(
        args=args, perf=combined, run_started=run_started, total_time=total_time,
        simulation_time=simulation_time, iterations=steps_done,
        n_stars_start=n_original, stars=final, stop_reason=stop_reason,
        implementation="mpi", n_ranks=size,
        n_threads=int(os.environ.get("OMP_NUM_THREADS", "1") or 1),
        load_imbalance=float(extra["load_imbalance"]))
    row.update(extra)
    serial.print_performance_summary(row)
    print(f"  communication {float(row['communication_time_s']):8.3f} s (slowest rank) · "
          f"sync {float(row['sync_time_s']):.3f} s · "
          f"load imbalance (forces max/mean) {float(row['load_imbalance']):.3f} · "
          f"slowest force time per cluster {row['cluster_forces_max_s']} s")
    if not args.no_results:
        serial.append_results(args.results, row)
        print(f"Added run '{row['run_name']}' to {args.results}")

    if (args.plot or live is not None) and matplotlib.get_backend().lower() != "agg":
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        # One rank failing would leave the others waiting in the next collective until
        # the walltime runs out. Print the error and take the whole job down instead.
        print(f"[rank {MPI.COMM_WORLD.Get_rank()}] unhandled error:", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
