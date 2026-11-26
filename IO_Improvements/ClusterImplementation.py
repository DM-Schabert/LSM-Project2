"""Cluster decomposition (v3): each group of ranks simulates one cluster of stars.

Cluster_Plan_v3.md, steps 2 and 3. The input is a CSV with a `cluster` column
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
    * Between clusters, --far-field monopole (default, step 3): every other
      cluster pulls on a star as one point mass at its centre of mass,
          a_i += G · Σ_{B ≠ own} M_B (R_B − r_i) / (|R_B − r_i|² + ε²)^1.5.
      Every cluster's mass, centre of mass and rms radius come from the step's
      one world Allreduce (below). The far field is computed after any merges,
      for the stars the rank owns then (merging keeps a cluster's mass and
      centre of mass).
      Theta_AB = (r_A + r_B) / |R_A − R_B|, with rms radii, says how good the
      approximation is: the largest theta of the run, and how many steps went
      over --theta-max, go in the printout and in results.csv.
    * --far-field off: no pull between clusters. Every cluster evolves as if it
      were alone, so its result is bit-identical to a serial or v1 run on that
      cluster's stars only (step 2's check, compare_clusters.py).
    * ONE collective per step (C4, 24 Nov). Right after its cluster's forces,
      before anything moves, each rank writes into its cluster's row of a small
      (C + 1, 7) table: the sums of its own stars (count, Σm, Σm·r, Σm·|r|²;
      far field on) and how many close pairs it found. The last row carries
      the stop requests (time limit, Ctrl+C, SIGTERM). One Allreduce(SUM) over
      the world then gives every rank the stop decision, its cluster's merge
      check and every cluster's summary. Before, these were up to three
      collectives per step. All clusters stop after the same step; a step that
      is told to stop ends before anything moves.
      Merging is still per cluster: a merge can bring stars close to others,
      so a cluster that merged checks again on its own communicator, and the
      other clusters never wait for that.
      With no merging and no far field, a step needs no exchange, and only
      every --stop-check-every steps has one.
      The step time includes the exchange, so it is everything a step costs
      (v1 leaves its stop check out of the step time).
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
ranks_per_cluster, far_field, cluster_hosts, cluster_forces_max_s (the
slowest rank's force time in each cluster, which shows load imbalance between
clusters, e.g. when only one of them merges), far_field_time_s,
far_interactions, theta_max and theta_steps_over. --save-final writes the
serial columns plus `cluster`.

Checks: compare_clusters.py (far field off: each cluster alone) and
cluster_reference.py (far field on: the same approximation without MPI, and
its force error against direct summation). validate_clusters.sh runs both.
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
from SerialImplementation import BoolArray, FloatArray, IntArray, Pair, Perf, Stars, Value, format_thousands

# v1's building blocks used here. If ParallelImplementation.py renames one, fail
# at once with a clear message instead of halfway through a job.
USED_FROM_PARALLEL: Tuple[str, ...] = (
    "ParallelState", "state_from_stars", "allgather_rows", "gather_full_state", "close_pairs_anywhere",
    "ParallelStopRequest", "block_accelerations", "cpu_model",
    "STOP_NONE", "STOP_INTERRUPTED", "STOP_WINDOW_CLOSED", "STOP_TIME_LIMIT", "STOP_REASONS")
_missing: List[str] = [name for name in USED_FROM_PARALLEL if not hasattr(pi, name)]
if _missing:
    raise ImportError(f"ParallelImplementation.py has no {', '.join(_missing)}, "
                      f"which ClusterImplementation.py needs")

# Every timed section, in a fixed order, so all ranks reduce the same vector (as in v1)
SECTIONS: Tuple[str, ...] = ("load", "distribute", "communication", "forces", "merge",
                             "integrate", "sync", "far_field", "plotting", "output")
BYTES_PER_ROW: int = 3 * 8  # one star's x, y, z as float64
SUMMARY_COLUMNS: int = 6    # count, Σm, Σm·x, Σm·y, Σm·z, Σm·|r|²
PAIRS_COLUMN: int = 6       # close pairs found, in the same table
TABLE_COLUMNS: int = 7
DEFAULT_THETA_MAX: float = 0.5


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
#
# At about 0.3 ms per step, the Python around each MPI call costs as much as the
# call itself (exchange_bench.py on the HPC: an Allreduce of one number took 5 µs,
# called through a helper 13 µs). So the per-step path allocates nothing,
# builds no lists, and times its sections by hand: a `with perf.time(...)`
# block costs about 3× more than two perf_counter() calls.

def record(perf: Perf, section: str, start: float) -> None:
    """Add the time since `start` to a section: what `with perf.time(section)` does, but cheaper."""
    perf.totals[section] += time.perf_counter() - start
    perf.calls[section] += 1


class PositionGather:
    """The per-step Allgatherv of the cluster's positions, as v1's allgather_rows.

    allgather_rows builds its count and offset lists on every call. Here the
    receive spec is built once and reused until the block layout changes,
    which happens only after a merge (state_from_stars makes a new Blocks).
    """

    def __init__(self) -> None:
        self._blocks: object = None
        self._out: Optional[FloatArray] = None
        self._spec: list = []
        self._received_bytes: int = 0

    def __call__(self, comm: MPI.Comm, state: "pi.ParallelState", perf: Perf) -> None:
        if state.blocks is not self._blocks or state.positions is not self._out:
            counts: List[int] = [int(c) * 3 for c in state.blocks.counts]  # in doubles: 3 per star
            displs: List[int] = [int(d) * 3 for d in state.blocks.displs]
            self._spec = [state.positions, counts, displs, MPI.DOUBLE]
            self._blocks, self._out = state.blocks, state.positions
            self._received_bytes = (state.count - state.my_count) * BYTES_PER_ROW
        comm.Allgatherv(state.my_positions, self._spec)
        perf.comm_bytes += self._received_bytes


def cluster_accelerations(state: "pi.ParallelState", comm: MPI.Comm, gather: PositionGather,
                          args: argparse.Namespace, perf: Perf) -> Tuple[FloatArray, List[Pair]]:
    """Collective (cluster): forces on the own block from every star of the cluster.

    The same code path as v1's allgather decomposition, on the cluster's
    communicator: with --tile-rows 1 it is bit-identical to the serial code.
    """
    assert state.positions is not None
    start_time: float = time.perf_counter()
    gather(comm, state, perf)
    record(perf, "communication", start_time)
    start, stop = state.my_range
    start_time = time.perf_counter()
    if args.tile_rows == 1:
        result: Tuple[FloatArray, List[Pair]] = serial.compute_accelerations(
            state.positions, state.masses, args.softening, args.merge_radius, start, stop)
    else:
        result = pi.block_accelerations(state.positions[start:stop], start, state.positions,
                                        state.masses, 0, args.softening, args.merge_radius,
                                        args.tile_rows)
    record(perf, "forces", start_time)
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


# ---------------------------------------------------------------- the far field (step 3)

@dataclass
class ClusterSummaries:
    """Every cluster's size, mass, centre of mass and rms radius, the same on every rank."""
    counts: IntArray       # (C,)
    masses: FloatArray     # (C,)   M_sun
    centres: FloatArray    # (C, 3) pc, centre of mass
    rms_radii: FloatArray  # (C,)   pc, mass-weighted rms distance from the centre of mass

    @classmethod
    def from_sums(cls, sums: FloatArray) -> "ClusterSummaries":
        """From a (C, 6) table of [count, Σm, Σm·x, Σm·y, Σm·z, Σm·|r|²] per cluster."""
        masses: FloatArray = sums[:, 1]
        centres: FloatArray = sums[:, 2:5] / masses[:, np.newaxis]
        mean_sq: FloatArray = sums[:, 5] / masses - np.sum(centres ** 2, axis=1)  # <|r|²> − |R|²
        return cls(counts=np.rint(sums[:, 0]).astype(np.int64), masses=masses, centres=centres,
                   rms_radii=np.sqrt(np.maximum(mean_sq, 0.0)))

    def theta(self) -> FloatArray:
        """(C, C) opening angles (r_A + r_B) / |R_A − R_B|; 0 on the diagonal."""
        distance: FloatArray = np.linalg.norm(self.centres[:, np.newaxis] - self.centres[np.newaxis], axis=2)
        np.fill_diagonal(distance, np.inf)
        return (self.rms_radii[:, np.newaxis] + self.rms_radii[np.newaxis]) / distance


def summary_row(positions: FloatArray, masses: FloatArray) -> FloatArray:
    """[count, Σm, Σm·x, Σm·y, Σm·z, Σm·|r|²] of the given stars (zeros for none)."""
    row: FloatArray = np.zeros(SUMMARY_COLUMNS, dtype=np.float64)
    row[0] = len(masses)
    row[1] = masses.sum()
    row[2:5] = masses @ positions
    row[5] = masses @ np.sum(positions ** 2, axis=1)
    return row


class StepExchanger:
    """The step's one world Allreduce: stop requests, merge check and cluster summaries.

    A (C + 1, 7) table: in its cluster's row, each rank writes the summary sums
    of its own stars (columns 0-5, only with the far field) and how many close
    pairs it found (column 6). The Allreduce(SUM) adds the rows of a cluster's
    ranks together. The last row counts stop requests per code (one-hot); the
    highest code anyone asked for wins, as with v1's Allreduce(MAX).

    Called after the force pass and before anything moves, so the summaries
    are those of the positions at the start of the step.

    The table and its result are allocated once and reused every step, and
    the result is read with plain Python, so a step pays for little more than
    the Allreduce itself.
    """

    def __init__(self, layout: ClusterLayout, world: MPI.Comm, with_summaries: bool) -> None:
        self.world: MPI.Comm = world
        self.with_summaries: bool = with_summaries
        self.n_clusters: int = layout.n_clusters
        self.table: FloatArray = np.zeros((layout.n_clusters + 1, TABLE_COLUMNS), dtype=np.float64)
        self.total: FloatArray = np.empty_like(self.table)
        self._my_row: FloatArray = self.table[layout.cluster_id]       # views: written in place
        self._stop_row: FloatArray = self.table[layout.n_clusters]
        self._my_total: FloatArray = self.total[layout.cluster_id]
        self._stop_total: FloatArray = self.total[layout.n_clusters]
        self._codes: List[int] = sorted(pi.STOP_REASONS, reverse=True)  # highest code first
        self.pairs_in_my_cluster: int = 0
        self.summaries: Optional[ClusterSummaries] = None

    def __call__(self, state: "pi.ParallelState", my_code: int, my_pair_count: int, perf: Perf) -> int:
        """Collective (world). Returns the agreed stop code; sets pairs_in_my_cluster and summaries."""
        start_time: float = time.perf_counter()
        self.table.fill(0.0)
        if self.with_summaries:
            start, stop = state.my_range
            self._my_row[:SUMMARY_COLUMNS] = summary_row(state.my_positions, state.masses[start:stop])
        self._my_row[PAIRS_COLUMN] = my_pair_count
        if my_code != pi.STOP_NONE:
            self._stop_row[my_code - 1] = 1.0
        self.world.Allreduce(self.table, self.total, op=MPI.SUM)
        stops: List[float] = self._stop_total.tolist()
        self.pairs_in_my_cluster = int(self._my_total[PAIRS_COLUMN])
        if self.with_summaries:  # a copy: the buffer is overwritten next step
            self.summaries = ClusterSummaries.from_sums(self.total[:self.n_clusters, :SUMMARY_COLUMNS].copy())
        record(perf, "sync", start_time)
        for code in self._codes:
            if stops[code - 1] > 0:
                return code
        return pi.STOP_NONE


def monopole_accelerations(positions: FloatArray, centres: FloatArray, cluster_masses: FloatArray,
                           own_cluster: int, softening: float) -> FloatArray:
    """Pull of every cluster except `own_cluster`, each as a point mass at its centre of mass.

    a_i = G · Σ_{B ≠ own} M_B (R_B − r_i) / (|R_B − r_i|² + ε²)^1.5, the same
    softened formula as the star-star force. Shared with cluster_reference.py.
    """
    others: BoolArray = np.arange(len(cluster_masses)) != own_cluster
    if not others.any() or len(positions) == 0:
        return np.zeros_like(positions)
    to_centres: FloatArray = centres[others][np.newaxis, :, :] - positions[:, np.newaxis, :]  # (n, C-1, 3)
    dist_sq: FloatArray = np.sum(to_centres ** 2, axis=2) + softening ** 2                   # (n, C-1)
    weights: FloatArray = cluster_masses[others][np.newaxis, :] * dist_sq ** -1.5
    return serial.G * np.einsum("nc,ncd->nd", weights, to_centres)


def far_field_accelerations(state: "pi.ParallelState", layout: ClusterLayout,
                            summaries: Optional[ClusterSummaries], args: argparse.Namespace,
                            perf: Perf) -> Optional[FloatArray]:
    """The pull of the other clusters on the own stars; None when there is no far field."""
    if summaries is None:
        return None
    start_time: float = time.perf_counter()
    far: FloatArray = monopole_accelerations(state.my_positions, summaries.centres, summaries.masses,
                                             layout.cluster_id, args.softening)
    record(perf, "far_field", start_time)
    return far


def cluster_step(state: "pi.ParallelState", layout: ClusterLayout, exchanger: StepExchanger,
                 gather: PositionGather, my_code: Optional[int], args: argparse.Namespace,
                 perf: Perf) -> Tuple["pi.ParallelState", int, int, Optional[ClusterSummaries]]:
    """One time step of this rank's cluster; same order as serial.step and v1.

    my_code: this rank's stop request when this step has the world exchange,
    None when it has none (no merging, no far field, and not a stop-check
    step). The step ends early, with nothing moved, when the exchange says
    stop. Returns the new state, how many stars of this cluster merged away,
    the agreed stop code, and the cluster summaries (far field on) or None.
    """
    comm: MPI.Comm = layout.comm
    n_before: int = state.count
    accelerations, my_pairs = cluster_accelerations(state, comm, gather, args, perf)

    summaries: Optional[ClusterSummaries] = None
    has_pairs: bool = False
    if my_code is not None:  # collective (world): stop requests + merge check + far-field summaries
        code: int = exchanger(state, my_code, len(my_pairs), perf)
        if code != pi.STOP_NONE:
            return state, 0, code, None  # stop before anything moves: the state is that of the last full step
        summaries = exchanger.summaries
        has_pairs = exchanger.pairs_in_my_cluster > 0

    # The first merge check came with the exchange. Any further ones (a merge can
    # bring stars within the merge radius of others) are per cluster: only the
    # clusters that merged take part, so the others never wait for them.
    while has_pairs:
        state = cluster_merge(state, my_pairs, comm, perf)
        accelerations, my_pairs = cluster_accelerations(state, comm, gather, args, perf)
        has_pairs = pi.close_pairs_anywhere(comm, my_pairs, perf)

    # After any merges: for the stars this rank owns now. Merging keeps every
    # cluster's mass and centre of mass, so the summaries are still valid.
    far: Optional[FloatArray] = far_field_accelerations(state, layout, summaries, args, perf)

    start_time: float = time.perf_counter()
    if far is not None:
        accelerations += far
    state.my_velocities += accelerations * args.dt
    state.my_positions += state.my_velocities * args.dt
    record(perf, "integrate", start_time)
    return state, n_before - state.count, pi.STOP_NONE, summaries


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


@dataclass
class ThetaLog:
    """The far field's opening angles over the run (kept on world rank 0)."""
    limit: float
    largest: float = 0.0
    steps_over: int = 0
    warned: bool = False

    def add(self, theta: FloatArray, step: int) -> None:
        worst: float = float(theta.max()) if theta.size else 0.0
        self.largest = max(self.largest, worst)
        if worst > self.limit:
            self.steps_over += 1
            if not self.warned:
                a, b = np.unravel_index(int(np.argmax(theta)), theta.shape)
                print(f"WARNING: step {format_thousands(step)}: theta = {worst:.2f} between clusters {a} and {b} "
                      f"is above --theta-max {self.limit:g}; the far field is less accurate for them "
                      f"(reported once; the count is in the summary)")
                self.warned = True


def reduce_cluster_perf(world: MPI.Comm, perf: Perf, layout: ClusterLayout, args: argparse.Namespace,
                        hosts_by_cluster: Optional[List[List[str]]], far_interactions: int,
                        theta_log: Optional[ThetaLog]) -> Tuple[Optional[Perf], Dict[str, Value]]:
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

    counters: IntArray = np.array([perf.interactions, perf.comm_bytes, far_interactions], dtype=np.int64)
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
        "far_field_time_s": float(t_max[SECTIONS.index("far_field")]),
        "far_interactions": int(counters_sum[2]),
        "theta_max": theta_log.largest if theta_log is not None else float("nan"),
        "theta_steps_over": theta_log.steps_over if theta_log is not None else 0,
        "theta_limit": args.theta_max,
    }
    for s, mn, mean in zip(SECTIONS, t_min, t_mean):
        if s in ("load", "plotting", "output", "far_field"):  # rank 0 only / tiny: min/mean add nothing
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
                        help="Agree on stopping (time limit / signals) every K steps. Only matters with "
                             "--merge-radius 0 and no far field: otherwise every step has the exchange "
                             "and the stop check comes with it for free (default: 1)")
    parser.add_argument("--tile-rows", type=non_negative_int, default=1, metavar="K",
                        help="Force kernel: K own stars per loop iteration (1 = the v1 kernel, "
                             "bit-identical to serial; 0 = automatic). Default: 1")
    # clusters
    parser.add_argument("--clusters", type=serial.positive_int, default=None, metavar="C",
                        help="Expected number of clusters; the run stops if the file has another number "
                             "(default: take it from the file)")
    parser.add_argument("--far-field", choices=("monopole", "off"), default="monopole",
                        help="Pull between clusters. monopole: every other cluster as a point mass at its "
                             "centre of mass (one Allreduce per step). off: none, every cluster evolves alone "
                             "(bit-identical to running each cluster on its own). Default: monopole")
    parser.add_argument("--theta-max", type=serial.non_negative_float, default=DEFAULT_THETA_MAX,
                        help="Warn when (r_A + r_B) / distance between two clusters goes above this; "
                             f"the run carries on (default: {DEFAULT_THETA_MAX:g})")
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

    # One world Allreduce per step carries the stop requests, the merge check and
    # (far field on) the cluster summaries. A step needs it when merging is on or
    # the far field is on; then stopping is checked every step for free. With
    # neither, only every --stop-check-every steps has the exchange.
    use_far_field: bool = args.far_field == "monopole" and n_clusters > 1
    exchange_every_step: bool = use_far_field or args.merge_radius > 0
    check_every: int = 1 if exchange_every_step else args.stop_check_every
    theta_log: Optional[ThetaLog] = ThetaLog(limit=args.theta_max) if use_far_field and is_root else None
    far_interactions: int = 0

    def my_stop_code() -> int:
        """This rank's stop request; only world rank 0's clock and window count, so there is one decision."""
        code: int = pi.STOP_INTERRUPTED if stop_request.requested else pi.STOP_NONE
        if is_root:
            if live is not None and not live.is_open():
                code = max(code, pi.STOP_WINDOW_CLOSED)
            expected_step_time: float = perf.step_times[-1] if perf.step_times else 0.0
            if time.perf_counter() + expected_step_time * check_every > deadline:
                code = max(code, pi.STOP_TIME_LIMIT)
        return code

    exchanger: StepExchanger = StepExchanger(layout, world, with_summaries=use_far_field)
    gather: PositionGather = PositionGather()

    world.Barrier()  # start the step clock together
    try:
        for i in range(1, args.steps + 1):
            # Everything a step costs is in its time, the exchange (and so the stop check) too
            step_start: float = time.perf_counter()
            has_exchange: bool = exchange_every_step or (i - 1) % check_every == 0
            state, n_merged, code, summaries = cluster_step(state, layout, exchanger, gather,
                                                            my_stop_code() if has_exchange else None,
                                                            args, perf)
            if code != pi.STOP_NONE:
                stop_reason = pi.STOP_REASONS[code]
                break
            perf.step_times.append(time.perf_counter() - step_start)
            if theta_log is not None and summaries is not None:
                theta_log.add(summaries.theta(), i)
            steps_done = i
            if use_far_field:
                far_interactions += state.my_count * (n_clusters - 1)
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
    combined, extra = reduce_cluster_perf(world, perf, layout, args, hosts_by_cluster,  # collective
                                          far_interactions, theta_log)
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
    print(f"Total momentum at end: {np.round(serial.total_momentum(final), 10)} M_sun·km/s (starts at 0; "
          f"the monopole far field does not conserve it exactly)")
    if theta_log is not None:
        print(f"Far field: largest theta {theta_log.largest:.3f} (limit {args.theta_max:g}), "
              f"above the limit in {theta_log.steps_over} of {steps_done} steps")
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
          f"slowest force time per cluster {row['cluster_forces_max_s']} s · "
          f"far field {float(row['far_field_time_s']):.3f} s")
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