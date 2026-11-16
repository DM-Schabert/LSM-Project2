"""MPI version of the direct-summation N-body simulation of a Gaia subset.

Same physics, same command line and same results.csv columns as
SerialImplementation.py. All physics (forces, merging), plotting and result
rows are imported from there, so serial and MPI run exactly the same kernel.
This file only adds distribution, communication and the stop logic.

Three decompositions, chosen with --decomposition:

    allgather (v1, default; weeks 1-6, plan sections 2.1-2.9)
      * Every rank OWNS a contiguous block of stars (block sizes differ by at
        most one, like np.array_split [W3]). It updates their velocities and
        positions only.
      * Positions are REPLICATED: at the start of every force evaluation all
        ranks Allgatherv their own block, so each rank holds all N positions
        and sums the force on its own stars from every star [W4 collectives].
        Allgatherv (not plain Allgather) lets N be any number, with no padding,
        so the result is bit-identical to the serial code (plan section 2.4,
        option B).

    ring (v2; week 7, plan section 8.1, iteration 7b)
      * The ranks form a 1-D PERIODIC CARTESIAN communicator (Create_cart) and
        Shift(0, 1) gives each rank its left and right neighbour [W7].
      * Each rank keeps only its own positions. A copy of its block TRAVELS
        around the ring: in P rounds every rank sees every block once, and adds
        the force that block exerts on its own stars. Positions per rank drop
        from N to about 2N/P (own block + the travelling buffers).
      * --ring-comm overlap (default): the next block is received (Irecv/Isend)
        while the current one is computed, then Waitall [W4 non-blocking].
        --ring-comm blocking: one Sendrecv after each block, for comparison.
      * The force on a star is summed block by block instead of in one sum, so
        the result is no longer bit-identical to the serial code. Validate
        against a tolerance (batch.sh MODE=validate does).

    2d (v3; week 7, plan section 8.1, iteration 7c; communication/2d_decomposition.html)
      * The ranks form a p_r x p_c CARTESIAN grid (Compute_dims, or --grid-rows),
        split into a row and a column communicator (Cartcomm.Sub) [W7 slide 3].
      * The N x N table of star pairs is cut into tiles: the stars are split once
        into p_r row blocks I and once into p_c column blocks J (both like
        np.array_split). Rank (r, c) computes the forces ON the stars of I_r FROM
        the stars of J_c: N^2/P pairs, as in v1.
      * Every rank of grid row r holds the positions and velocities of I_r (p_c
        identical copies) and integrates them. Each step:
          1. column: Allgatherv of the positions of J_c; every rank of the column
             contributes the part of J_c that lies in its own I_r
          2. forces of J_c on I_r (block_accelerations, global indices)
          3. row: Allreduce(SUM) of the partial forces on I_r
          4. kick + drift of I_r
      * Received per rank and step: about N/p_c positions plus N/p_r forces,
        instead of N positions. Positions held: N/p_r + N/p_c.
      * Merging is not supported (--merge-radius 0). The p_c copies of I_r stay
        identical as long as the Allreduce gives every rank the same sum; the
        run checks that at the end (grid_copy_max_diff in results.csv).

Shared by all three:
    * Masses, classes, member counts and the owner map are replicated. They
      only change in a merge, which every rank performs identically.
    * Merging: the decision "are there close pairs anywhere?" is a collective
      Allreduce, so all ranks enter the merge loop together. The pair lists,
      positions and velocities are then gathered to every rank, and every rank
      runs the serial merge_stars on identical data (plan section 2.5).
    * Stopping: time limit, Ctrl+C/SIGTERM and closing the live-plot window
      are only *local requests*. Ranks stop when an Allreduce(MAX) of those
      requests says so, so they always stop after the same step. Otherwise
      the next collective would deadlock [W1].
    * Hot-path communication uses the upper-case buffer API (Allgatherv,
      Isend/Irecv, Allreduce, Bcast on NumPy arrays). Lower-case (pickled)
      calls are used only for rare, small Python objects: pair lists, classes,
      error messages [W4: case sensitivity].
    * --tile-rows K: compute the forces of K own stars at once (2-D broadcasting)
      instead of one star per Python loop iteration. 1 (default) is the v1
      kernel. 0 picks K so the temporary arrays stay around 8 MB. In ring mode
      a rank loops over its rows once per block, so with K = 1 the Python
      loop overhead is NOT divided by P; measure ring with and without tiling.

Usage (same options as SerialImplementation.py, plus the MPI-only ones):
    mpirun -np 4 python3 ParallelImplementation.py --quick
    mpirun -np 4 python3 ParallelImplementation.py --data gaia_subsets/gaia_random_5000.csv --steps 200
    mpirun -np 4 python3 ParallelImplementation.py --decomposition ring --tile-rows 0
    mpirun -np 6 python3 ParallelImplementation.py --decomposition 2d --merge-radius 0 --tile-rows 0
    mpirun -np 4 python3 ParallelImplementation.py --merge-radius 0 --save-final final_mpi.csv
    mpirun --oversubscribe -np 8 python3 ParallelImplementation.py ...  # laptop: correctness only, never timing!

Validation: run the serial and the MPI version with the same options and
--save-final, then compare the CSVs. allgather without tiling should match to
the last digit (final_position_checksum in results.csv as well); ring and
tiled runs should match to a small relative tolerance.

Timing (per rank, then reduced to rank 0; results.csv shows the MAX over ranks):
    load           rank 0 reading the CSV
    distribute     broadcasting the initial state to all ranks
    communication  allgather: the Allgatherv of positions before every force evaluation
                   ring: the exposed ring traffic (posting + Waitall, or Sendrecv)
                   2d: the column Allgatherv plus the row Allreduce
    forces         the force kernel on the own block
    merge          the whole merge path (its gathers included)
    integrate      kick + drift of the own block
    sync           stop-flag and pair-count Allreduces
    plotting / output   rank 0 (+ the gathers they need)
"""

import argparse
import hashlib
import os
import platform
import signal
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpi4py import MPI

import SerialImplementation as serial
from SerialImplementation import (BoolArray, FloatArray, IntArray, Pair, Perf, Stars, StrArray,
                                  Value, format_thousands)

# Stop codes. Ranks agree on the largest one with Allreduce(MAX).
STOP_NONE: int = 0
STOP_INTERRUPTED: int = 1
STOP_WINDOW_CLOSED: int = 2
STOP_TIME_LIMIT: int = 3
STOP_REASONS: Dict[int, str] = {STOP_INTERRUPTED: "interrupted",
                                STOP_WINDOW_CLOSED: "window_closed",
                                STOP_TIME_LIMIT: "time_limit"}

# Every timed section, in a fixed order, so all ranks reduce the same vector
SECTIONS: Tuple[str, ...] = ("load", "distribute", "communication", "forces", "merge",
                             "integrate", "sync", "plotting", "output")

BYTES_PER_ROW: int = 3 * 8  # one star's x, y, z (or vx, vy, vz) as float64
RING_TAG: int = 7           # tag of the travelling position blocks
TILE_TARGET_ELEMENTS: int = 1 << 18  # --tile-rows 0: ~2^18 pairs per tile (~8 MB of temporaries)
NDArray3 = np.ndarray                # (t, n_j, 3) temporaries in the tiled kernel


# ---------------------------------------------------------------- distribution

@dataclass
class Blocks:
    """Which stars each rank owns: rank r owns [displs[r], displs[r] + counts[r])."""
    counts: IntArray  # (P,) stars per rank
    displs: IntArray  # (P,) index of each rank's first star

    def range_of(self, rank: int) -> Tuple[int, int]:
        start: int = int(self.displs[rank])
        return start, start + int(self.counts[rank])


def make_blocks(n_stars: int, n_ranks: int) -> Blocks:
    """Contiguous blocks with sizes that differ by at most one (same split as np.array_split).

    With fewer stars than ranks, the last ranks own nothing. That is allowed:
    they still take part in every collective with an empty block.
    """
    counts: IntArray = np.full(n_ranks, n_stars // n_ranks, dtype=np.int64)
    counts[: n_stars % n_ranks] += 1
    displs: IntArray = np.zeros(n_ranks, dtype=np.int64)
    displs[1:] = np.cumsum(counts)[:-1]
    return Blocks(counts=counts, displs=displs)


@dataclass
class ParallelState:
    """The simulation state as seen by one rank.

    allgather: positions is (N, 3) on every rank, but only valid right after
    an allgather: each rank moves its own stars in my_positions, and the next
    force evaluation gathers them again.
    ring: positions is None during the run (only own positions are kept);
    the merge path gathers them into a temporary array when it needs them.
    """
    positions: Optional[FloatArray]  # (N, 3) replicated (allgather), None (ring)
    my_positions: FloatArray   # (n_own, 3)  owned
    my_velocities: FloatArray  # (n_own, 3)  owned
    masses: FloatArray         # (N,)        replicated
    members: IntArray          # (N,)        replicated
    owner: IntArray            # (N_original,) replicated
    sptypes: StrArray          # (N,)        replicated
    sptype_guessed: BoolArray  # (N,)        replicated
    blocks: Blocks
    rank: int

    @property
    def count(self) -> int:
        return len(self.masses)

    @property
    def my_range(self) -> Tuple[int, int]:
        return self.blocks.range_of(self.rank)

    @property
    def my_count(self) -> int:
        return int(self.blocks.counts[self.rank])


def state_from_stars(stars: Stars, comm: MPI.Comm, keep_all_positions: bool = True) -> ParallelState:
    """Split a full, identical-on-every-rank Stars object into this rank's state.

    keep_all_positions=False (ring) keeps only the own block of positions.
    """
    blocks: Blocks = make_blocks(stars.count, comm.Get_size())
    start, stop = blocks.range_of(comm.Get_rank())
    return ParallelState(
        positions=np.ascontiguousarray(stars.positions, dtype=np.float64) if keep_all_positions else None,
        # Copies, never views: a view would alias the Allgatherv receive buffer
        my_positions=np.array(stars.positions[start:stop], dtype=np.float64, order="C"),
        my_velocities=np.array(stars.velocities[start:stop], dtype=np.float64, order="C"),
        masses=stars.masses, members=stars.members, owner=stars.owner,
        sptypes=stars.sptypes, sptype_guessed=stars.sptype_guessed,
        blocks=blocks, rank=comm.Get_rank())


def allgather_rows(comm: MPI.Comm, mine: FloatArray, out: FloatArray, blocks: Blocks,
                   perf: Perf) -> None:
    """Gather every rank's (n_own, 3) block into the full (N, 3) array `out` on every rank.

    Counts and displacements are in doubles (3 per star), because the
    datatype is MPI.DOUBLE. Week 8 can replace this with a custom
    "xyz" datatype and count in stars instead.
    """
    counts: List[int] = [int(c) * 3 for c in blocks.counts]
    displs: List[int] = [int(d) * 3 for d in blocks.displs]
    comm.Allgatherv(mine, [out, counts, displs, MPI.DOUBLE])
    perf.comm_bytes += (len(out) - len(mine)) * BYTES_PER_ROW  # bytes this rank received


def gather_full_state(state: ParallelState, comm: MPI.Comm, perf: Perf) -> Stars:
    """Collective: every rank gets the complete current state (for plots and the final save)."""
    positions: FloatArray = np.empty((state.count, 3), dtype=np.float64)
    velocities: FloatArray = np.empty((state.count, 3), dtype=np.float64)
    allgather_rows(comm, state.my_positions, positions, state.blocks, perf)
    allgather_rows(comm, state.my_velocities, velocities, state.blocks, perf)
    return Stars(positions=positions, velocities=velocities, masses=state.masses,
                 members=state.members, owner=state.owner,
                 sptypes=state.sptypes, sptype_guessed=state.sptype_guessed)


# ---------------------------------------------------------------- the ring (week 7)

class Ring:
    """The 1-D periodic Cartesian communicator and the two travelling buffers.

    Rank r sends to its right neighbour and receives from its left one. After
    s rounds it holds the block that rank (r - s) mod P owns, so after P - 1
    rounds it has seen every block exactly once.
    """

    def __init__(self, cart: MPI.Cartcomm, overlap: bool) -> None:
        self.cart: MPI.Cartcomm = cart
        self.left, self.right = cart.Shift(0, 1)  # (source, dest) for a shift of +1 [W7]
        self.overlap: bool = overlap
        self._buffers: Tuple[FloatArray, FloatArray] = (np.empty((0, 3)), np.empty((0, 3)))

    def buffers(self, max_rows: int) -> Tuple[FloatArray, FloatArray]:
        """Two (max_rows, 3) buffers, reallocated only when N changes (after a merge)."""
        if len(self._buffers[0]) != max_rows:
            self._buffers = (np.empty((max_rows, 3), dtype=np.float64),
                             np.empty((max_rows, 3), dtype=np.float64))
        return self._buffers


def tile_size(tile_rows: int, n_j: int) -> int:
    """Rows per tile: tile_rows as given, or (0) as many as keep a tile near TILE_TARGET_ELEMENTS pairs."""
    if tile_rows > 0:
        return tile_rows
    return max(1, TILE_TARGET_ELEMENTS // max(n_j, 1))


def block_accelerations(i_positions: FloatArray, i_start: int, j_positions: FloatArray,
                        j_masses: FloatArray, j_start: int, softening: float, merge_radius: float,
                        tile_rows: int = 1) -> Tuple[FloatArray, List[Pair]]:
    """Acceleration on stars i_start.. (rows of i_positions) from the stars j_start.. (rows of j_positions).

    The same formula as serial.compute_accelerations, restricted to one block
    of j: a_i += G · Σ_j m_j (r_j − r_i) / (|r_j − r_i|² + ε²)^1.5. Indices are
    global, so the self-force (i == j) is excluded and close pairs (i, j),
    j > i, are found exactly once, by the owner of i, whichever block j is in.

    tile_rows = 1 loops over i one star at a time, exactly like the serial
    kernel. tile_rows > 1 (or 0 = automatic) handles a tile of i rows per
    loop iteration with 2-D broadcasting: fewer Python iterations, larger
    temporaries.
    """
    n_i: int = len(i_positions)
    n_j: int = len(j_positions)
    accelerations: FloatArray = np.zeros((n_i, 3), dtype=np.float64)
    close_pairs: List[Pair] = []
    if n_i == 0 or n_j == 0:
        return accelerations, close_pairs
    softening_sq: float = softening ** 2
    merge_radius_sq: float = merge_radius ** 2
    j_stop: int = j_start + n_j
    tile: int = tile_size(tile_rows, n_j)

    if tile == 1:
        for li in range(n_i):
            i: int = i_start + li
            dis_vectors: FloatArray = j_positions - i_positions[li]          # from star i to every j in the block
            raw_dist_sq: FloatArray = np.sum(dis_vectors ** 2, axis=1)
            if merge_radius > 0:
                first: int = max(i + 1 - j_start, 0)                         # only j > i
                close_j: IntArray = np.nonzero(raw_dist_sq[first:] < merge_radius_sq)[0] + first + j_start
                close_pairs.extend((i, int(j)) for j in close_j)
            dist_sq: FloatArray = raw_dist_sq + softening_sq
            if j_start <= i < j_stop:
                dist_sq[i - j_start] = np.inf                                # no force from itself
            inv_dist_cubed: FloatArray = dist_sq ** -1.5
            accelerations[li] = serial.G * np.sum((j_masses * inv_dist_cubed)[:, np.newaxis] * dis_vectors,
                                                  axis=0)
        return accelerations, close_pairs

    j_index: IntArray = np.arange(j_start, j_stop)
    for t0 in range(0, n_i, tile):
        t1: int = min(n_i, t0 + tile)
        i_index: IntArray = np.arange(i_start + t0, i_start + t1)
        dis: NDArray3 = j_positions[np.newaxis, :, :] - i_positions[t0:t1, np.newaxis, :]  # (t, n_j, 3)
        raw: FloatArray = np.sum(dis ** 2, axis=2)                                         # (t, n_j)
        if merge_radius > 0:
            rows, cols = np.nonzero((raw < merge_radius_sq) & (j_index[np.newaxis, :] > i_index[:, np.newaxis]))
            close_pairs.extend((int(i_index[r]), int(j_index[c])) for r, c in zip(rows, cols))
        dist_sq = raw + softening_sq
        own: BoolArray = (i_index >= j_start) & (i_index < j_stop)
        if own.any():
            dist_sq[np.nonzero(own)[0], i_index[own] - j_start] = np.inf
        weights: FloatArray = j_masses[np.newaxis, :] * dist_sq ** -1.5                    # (t, n_j)
        accelerations[t0:t1] = serial.G * np.einsum("tj,tjc->tc", weights, dis)
    return accelerations, close_pairs



def ring_accelerations(state: ParallelState, ring: Ring, args: argparse.Namespace,
                       perf: Perf) -> Tuple[FloatArray, List[Pair]]:
    """Collective (over the ring): forces on the own block from all N stars, one travelling block at a time."""
    comm: MPI.Cartcomm = ring.cart
    size: int = comm.Get_size()
    counts, displs = state.blocks.counts, state.blocks.displs
    my_start, _ = state.my_range
    current, upcoming = ring.buffers(int(counts.max()) if len(counts) else 0)
    current[: state.my_count] = state.my_positions        # the copy of our block that will travel
    origin: int = state.rank                               # whose block is in `current`

    accelerations: FloatArray = np.zeros((state.my_count, 3), dtype=np.float64)
    pairs: List[Pair] = []
    for round_ in range(size):
        n_current: int = int(counts[origin])
        next_origin: int = (origin - 1) % size
        n_upcoming: int = int(counts[next_origin])
        last: bool = round_ == size - 1
        requests: List[MPI.Request] = []
        if not last and ring.overlap:
            with perf.time("communication"):  # start moving the block on while we compute with it
                requests = [comm.Irecv(upcoming[:n_upcoming], source=ring.left, tag=RING_TAG),
                            comm.Isend(current[:n_current], dest=ring.right, tag=RING_TAG)]
        with perf.time("forces"):
            j0: int = int(displs[origin])
            block_acc, block_pairs = block_accelerations(
                state.my_positions, my_start, current[:n_current], state.masses[j0:j0 + n_current], j0,
                args.softening, args.merge_radius, args.tile_rows)
            accelerations += block_acc
            pairs.extend(block_pairs)
        if not last:
            with perf.time("communication"):
                if ring.overlap:
                    MPI.Request.Waitall(requests)
                else:
                    comm.Sendrecv(current[:n_current], dest=ring.right, sendtag=RING_TAG,
                                  recvbuf=upcoming[:n_upcoming], source=ring.left, recvtag=RING_TAG)
            perf.comm_bytes += n_upcoming * BYTES_PER_ROW  # bytes this rank received
            current, upcoming = upcoming, current
            origin = next_origin
    return accelerations, pairs


# ---------------------------------------------------------------- the 2-D force decomposition (week 7)

def grid_dims(size: int, grid_rows: int) -> List[int]:
    """[p_r, p_c] for P ranks: Compute_dims (as square as possible, p_r >= p_c), or p_r = grid_rows."""
    if grid_rows == 0:
        return list(MPI.Compute_dims(size, 2))
    if size % grid_rows != 0:
        raise ValueError(f"--grid-rows {grid_rows} does not divide the {size} ranks")
    return [grid_rows, size // grid_rows]


class Grid2D:
    """The p_r x p_c Cartesian grid, its row and column communicators, and the column exchange.

    Rank (r, c) computes the forces on the row block I_r from the column block J_c.
    Every rank of grid row r holds the positions and velocities of I_r. The
    positions of J_c are assembled by an Allgatherv in the grid column: the rank of
    grid row k contributes J_c ∩ I_k (rows partition the stars, so these pieces
    are contiguous and in order of k, and together they are exactly J_c).
    """

    def __init__(self, cart: MPI.Cartcomm) -> None:
        self.cart: MPI.Cartcomm = cart
        self.pr, self.pc = (int(d) for d in cart.Get_topo()[0])
        self.r, self.c = (int(x) for x in cart.Get_coords(cart.Get_rank()))
        self.row: MPI.Cartcomm = cart.Sub([False, True])  # same grid row r; its ranks are the columns c
        self.col: MPI.Cartcomm = cart.Sub([True, False])  # same grid column c; its ranks are the rows r
        self.set_count(0)

    def set_count(self, n_stars: int) -> None:
        """Blocks and the column exchange for N stars (called once N is known)."""
        self.rows: Blocks = make_blocks(n_stars, self.pr)   # I blocks
        self.cols: Blocks = make_blocks(n_stars, self.pc)   # J blocks
        i0, i1 = self.rows.range_of(self.r)
        j0, j1 = self.cols.range_of(self.c)
        counts: List[int] = []
        displs: List[int] = []
        for k in range(self.pr):
            a, b = self.rows.range_of(k)
            lo, hi = max(a, j0), min(b, j1)
            n_k: int = max(0, hi - lo)
            counts.append(3 * n_k)                            # in doubles, like allgather_rows
            displs.append(3 * (lo - j0) if n_k else 0)
        self.col_counts: List[int] = counts
        self.col_displs: List[int] = displs
        lo, hi = max(i0, j0), min(i1, j1)
        self.piece: Tuple[int, int] = (lo - i0, hi - i0) if hi > lo else (0, 0)  # my part of J_c, local in I_r
        self.pos_j: FloatArray = np.empty((j1 - j0, 3), dtype=np.float64)
        self.acc_partial: FloatArray = np.empty((i1 - i0, 3), dtype=np.float64)

    def shape(self) -> str:
        return f"{self.pr}x{self.pc}"

    def free(self) -> None:
        self.row.Free()
        self.col.Free()


def state_for_grid(stars: Stars, grid: Grid2D) -> ParallelState:
    """This rank's state for the 2-D decomposition: the row block I_r (shared by its grid row).

    blocks are the p_r row blocks and rank is the grid row, so my_range/my_count are I_r.
    The full positions are kept for start-up only (main drops them, like the ring).
    """
    grid.set_count(stars.count)
    start, stop = grid.rows.range_of(grid.r)
    return ParallelState(
        positions=np.ascontiguousarray(stars.positions, dtype=np.float64),
        my_positions=np.array(stars.positions[start:stop], dtype=np.float64, order="C"),
        my_velocities=np.array(stars.velocities[start:stop], dtype=np.float64, order="C"),
        masses=stars.masses, members=stars.members, owner=stars.owner,
        sptypes=stars.sptypes, sptype_guessed=stars.sptype_guessed,
        blocks=grid.rows, rank=grid.r)


def grid_accelerations(state: ParallelState, grid: Grid2D, args: argparse.Namespace,
                       perf: Perf) -> Tuple[FloatArray, List[Pair]]:
    """Collective (over the grid): forces on I_r from all N stars, in three steps."""
    i0, i1 = state.my_range
    j0, j1 = grid.cols.range_of(grid.c)
    a, b = grid.piece
    with perf.time("communication"):   # 1. column: assemble the positions of J_c
        grid.col.Allgatherv(state.my_positions[a:b], [grid.pos_j, grid.col_counts, grid.col_displs, MPI.DOUBLE])
    with perf.time("forces"):          # 2. forces of J_c on I_r
        partial, _ = block_accelerations(state.my_positions, i0, grid.pos_j, state.masses[j0:j1], j0,
                                         args.softening, 0.0, args.tile_rows)
    accelerations: FloatArray = np.empty_like(partial)
    with perf.time("communication"):   # 3. row: sum the partial forces of the p_c column blocks
        grid.row.Allreduce(partial, accelerations, op=MPI.SUM)
    overlap: int = max(0, min(i1, j1) - max(i0, j0))
    perf.interactions += (i1 - i0) * (j1 - j0) - overlap          # pairs of this tile, self-force excluded
    perf.comm_bytes += ((j1 - j0) - (b - a) + ((i1 - i0) if grid.pc > 1 else 0)) * BYTES_PER_ROW  # approx.
    return accelerations, []


def gather_grid_state(state: ParallelState, grid: Grid2D, comm: MPI.Comm, perf: Perf) -> Stars:
    """Collective: the full state on every rank. Only grid column 0 sends; the other
    columns hold identical copies of their row block and send nothing."""
    n: int = state.count
    counts: List[int] = []
    displs: List[int] = []
    for k in range(comm.Get_size()):
        r_k, c_k = divmod(k, grid.pc)                     # row-major ranks (reorder=False)
        start, stop = grid.rows.range_of(r_k)
        counts.append(3 * (stop - start) if c_k == 0 else 0)
        displs.append(3 * start)
    mine: slice = slice(None) if grid.c == 0 else slice(0, 0)
    positions: FloatArray = np.empty((n, 3), dtype=np.float64)
    velocities: FloatArray = np.empty((n, 3), dtype=np.float64)
    comm.Allgatherv(state.my_positions[mine], [positions, counts, displs, MPI.DOUBLE])
    comm.Allgatherv(state.my_velocities[mine], [velocities, counts, displs, MPI.DOUBLE])
    perf.comm_bytes += 2 * (n - (state.my_count if grid.c == 0 else 0)) * BYTES_PER_ROW
    return Stars(positions=positions, velocities=velocities, masses=state.masses,
                 members=state.members, owner=state.owner,
                 sptypes=state.sptypes, sptype_guessed=state.sptype_guessed)


def grid_copy_difference(state: ParallelState, grid: Grid2D, comm: MPI.Comm) -> float:
    """Collective: the largest difference between a rank's copy of I_r and the copy of column 0.

    0 means the row copies stayed identical (every rank got the same Allreduce sum).
    """
    reference: FloatArray = np.concatenate([state.my_positions, state.my_velocities])
    if grid.c != 0:
        reference = np.empty_like(reference)
    grid.row.Bcast(reference, root=0)
    mine: FloatArray = np.concatenate([state.my_positions, state.my_velocities])
    local: FloatArray = np.array([float(np.max(np.abs(mine - reference))) if mine.size else 0.0])
    result: FloatArray = np.empty(1, dtype=np.float64)
    comm.Allreduce(local, result, op=MPI.MAX)
    return float(result[0])


# ---------------------------------------------------------------- start-up

def load_and_distribute(args: argparse.Namespace, comm: MPI.Comm, perf: Perf,
                        grid: Optional[Grid2D] = None) -> ParallelState:
    """Rank 0 reads the CSV (and draws the guessed classes); everyone else gets it by Bcast.

    Only one rank touches the file system (plan section 2.6). Large arrays go
    with upper-case Bcast, the class letters once with lower-case bcast.
    """
    rank: int = comm.Get_rank()
    stars: Stars | None = None
    error: str | None = None

    with perf.time("load"):
        if rank == 0:
            try:
                stars = serial.load_subset(args.data, np.random.default_rng(args.seed))
            except Exception as exc:  # tell the other ranks instead of leaving them waiting
                error = f"{type(exc).__name__}: {exc}"

    with perf.time("distribute"):
        error = comm.bcast(error, root=0)
        if error is not None:
            if rank == 0:
                print(f"Could not load the data: {error}", file=sys.stderr)
            raise SystemExit(1)  # every rank exits, nobody is left waiting

        n: int = comm.bcast(stars.count if rank == 0 else None, root=0)
        if rank != 0:
            stars = Stars(positions=np.empty((n, 3), dtype=np.float64),
                          velocities=np.empty((n, 3), dtype=np.float64),
                          masses=np.empty(n, dtype=np.float64),
                          members=np.empty(n, dtype=np.int64),
                          owner=np.empty(n, dtype=np.int64),
                          sptypes=np.empty(n, dtype="<U1"),
                          sptype_guessed=np.empty(n, dtype=np.bool_))
        assert stars is not None
        if rank == 0:
            # Bcast sends raw memory: a column-major (Fortran-ordered) array would arrive
            # scrambled on ranks that receive into row-major (C-ordered) buffers.
            # Force C order before sending.
            stars.positions = np.ascontiguousarray(stars.positions)
            stars.velocities = np.ascontiguousarray(stars.velocities)
        for array in (stars.positions, stars.velocities, stars.masses, stars.members, stars.owner):
            comm.Bcast(array, root=0)
        # Small, one-off, not numeric: lower-case is fine here
        stars.sptypes, stars.sptype_guessed = comm.bcast(
            (stars.sptypes, stars.sptype_guessed) if rank == 0 else None, root=0)
        perf.comm_bytes += 0 if rank == 0 else n * (2 * BYTES_PER_ROW + 3 * 8)

    if grid is not None:
        return state_for_grid(stars, grid)
    return state_from_stars(stars, comm)  # full positions at start-up; ring and 2d drop them in main


# ---------------------------------------------------------------- physics (parallel parts)

def parallel_accelerations(state: ParallelState, comm: MPI.Comm, ring: Optional[Ring],
                           args: argparse.Namespace, perf: Perf,
                           grid: Optional[Grid2D] = None) -> Tuple[FloatArray, List[Pair]]:
    """Collective: forces on the own block from all N stars (allgather, ring or 2d)."""
    if grid is not None:
        return grid_accelerations(state, grid, args, perf)  # counts its own interactions (one tile)
    if ring is not None:
        result: Tuple[FloatArray, List[Pair]] = ring_accelerations(state, ring, args, perf)
    else:
        assert state.positions is not None
        with perf.time("communication"):
            allgather_rows(comm, state.my_positions, state.positions, state.blocks, perf)
        start, stop = state.my_range
        with perf.time("forces"):
            if args.tile_rows == 1:  # the v1 kernel, bit-identical to the serial code
                result = serial.compute_accelerations(
                    state.positions, state.masses, args.softening, args.merge_radius, start, stop)
            else:
                result = block_accelerations(state.positions[start:stop], start, state.positions,
                                             state.masses, 0, args.softening, args.merge_radius,
                                             args.tile_rows)
    perf.interactions += state.my_count * (state.count - 1)  # own rows only; summed over ranks later
    return result


def close_pairs_anywhere(comm: MPI.Comm, my_pairs: List[Pair], perf: Perf) -> bool:
    """Collective: does ANY rank have a close pair? Every rank gets the same answer.

    In the serial code `while close_pairs:` is a local test. Here a local
    test would let one rank merge while the others carry on, and the next
    Allgatherv would deadlock.
    """
    with perf.time("sync"):
        local: IntArray = np.array([len(my_pairs)], dtype=np.int64)
        total: IntArray = np.empty(1, dtype=np.int64)
        comm.Allreduce(local, total, op=MPI.SUM)
    return int(total[0]) > 0


def parallel_merge(state: ParallelState, my_pairs: List[Pair], comm: MPI.Comm, ring: Optional[Ring],
                   perf: Perf) -> ParallelState:
    """Collective: give every rank the full state and all pairs, then run the serial merge everywhere.

    Each pair (i, j), i < j, was found exactly once, by the owner of i.
    Joining the lists in rank order gives the serial pair list (allgather) or
    the same pairs in another order (ring); merge_stars does not depend on the
    order. Merging is a rare event, so the slow-but-simple gather of
    everything is acceptable.
    """
    with perf.time("merge"):
        all_pairs: List[Pair] = [pair for rank_pairs in comm.allgather(my_pairs) for pair in rank_pairs]
        perf.comm_bytes += (len(all_pairs) - len(my_pairs)) * 16  # approx.: two int64 per pair
        if ring is None:
            assert state.positions is not None
            positions: FloatArray = state.positions  # still current: gathered by this step's force pass
        else:
            positions = np.empty((state.count, 3), dtype=np.float64)  # the ring keeps no full copy
            allgather_rows(comm, state.my_positions, positions, state.blocks, perf)
        velocities: FloatArray = np.empty((state.count, 3), dtype=np.float64)
        allgather_rows(comm, state.my_velocities, velocities, state.blocks, perf)
        full: Stars = Stars(positions=positions, velocities=velocities, masses=state.masses,
                            members=state.members, owner=state.owner,
                            sptypes=state.sptypes, sptype_guessed=state.sptype_guessed)
        merged: Stars = serial.merge_stars(full, all_pairs)  # identical inputs -> identical result
        return state_from_stars(merged, comm, keep_all_positions=ring is None)  # new N -> new blocks


def parallel_step(state: ParallelState, comm: MPI.Comm, ring: Optional[Ring], args: argparse.Namespace,
                  perf: Perf, grid: Optional[Grid2D] = None) -> Tuple[ParallelState, int]:
    """One time step; same order as serial.step: forces -> (merge -> forces)* -> kick -> drift.

    2d: merging is off (checked in main), so the merge loop never runs; every rank of a
    grid row integrates the same row block with the same summed forces."""
    n_before: int = state.count
    accelerations, my_pairs = parallel_accelerations(state, comm, ring, args, perf, grid)

    # Every rank knows args.merge_radius, so skipping the check when merging
    # is off is the same decision on every rank.
    if args.merge_radius > 0:
        while close_pairs_anywhere(comm, my_pairs, perf):
            state = parallel_merge(state, my_pairs, comm, ring, perf)
            accelerations, my_pairs = parallel_accelerations(state, comm, ring, args, perf)

    with perf.time("integrate"):
        state.my_velocities += accelerations * args.dt
        state.my_positions += state.my_velocities * args.dt
    return state, n_before - state.count


# ---------------------------------------------------------------- stopping together

class ParallelStopRequest(serial.StopRequest):
    """Ctrl+C / SIGTERM only *request* a stop. They never stop this rank on its own.

    The serial version raises KeyboardInterrupt on a second Ctrl+C. Here that
    could leave one rank in the middle of a collective and the others waiting
    forever, so repeated signals are ignored and the stop goes through
    agree_on_stop like everything else.
    """

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank: int = rank

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if not self.requested and self.rank == 0:
            print(f"\nReceived {signal.Signals(signum).name}: all ranks stop after this step and save")
        self.requested = True


def agree_on_stop(comm: MPI.Comm, my_code: int, perf: Perf) -> int:
    """Collective: the largest stop code over all ranks (0 = carry on)."""
    with perf.time("sync"):
        local: IntArray = np.array([my_code], dtype=np.int64)
        result: IntArray = np.empty(1, dtype=np.int64)
        comm.Allreduce(local, result, op=MPI.MAX)
    return int(result[0])


# ---------------------------------------------------------------- results

def cpu_model() -> str:
    """CPU model name, which goes in the report [Project 2 notes]."""
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def code_fingerprint() -> str:
    """Short hash of the two source files, so every results row can be traced to a code version."""
    digest = hashlib.md5()
    for name in ("ParallelImplementation.py", "SerialImplementation.py"):
        path: Path = Path(__file__).resolve().parent / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def decomposition_name(args: argparse.Namespace, grid: Optional[Grid2D] = None) -> str:
    if args.decomposition == "2d":
        return f"2d_{grid.shape()}" if grid is not None else "2d"
    return "allgather_block" if args.decomposition == "allgather" else f"ring_{args.ring_comm}"


def reduce_perf(comm: MPI.Comm, perf: Perf, args: argparse.Namespace,
                grid: Optional[Grid2D] = None) -> Tuple[Perf | None, Dict[str, Value]]:
    """Collective: combine every rank's timings on rank 0.

    Returns, on rank 0 only, a Perf whose section times and step times are the
    MAX over ranks (the slowest rank decides when a step is done), whose
    interaction count and comm_bytes are SUMs, and extra columns for
    results.csv (min/mean per section, hosts, memory). Other ranks get (None, {}).
    """
    rank: int = comm.Get_rank()
    totals: FloatArray = np.array([perf.totals.get(s, 0.0) for s in SECTIONS], dtype=np.float64)
    t_max, t_min, t_sum = (np.empty_like(totals) for _ in range(3))
    comm.Reduce(totals, t_max, op=MPI.MAX, root=0)
    comm.Reduce(totals, t_min, op=MPI.MIN, root=0)
    comm.Reduce(totals, t_sum, op=MPI.SUM, root=0)

    # Every rank did the same number of steps (they stop together), so these line up
    steps: FloatArray = np.array(perf.step_times, dtype=np.float64)
    steps_max: FloatArray = np.empty_like(steps)
    comm.Reduce(steps, steps_max, op=MPI.MAX, root=0)

    counters: IntArray = np.array([perf.interactions, perf.comm_bytes], dtype=np.int64)
    counters_sum: IntArray = np.empty_like(counters)
    comm.Reduce(counters, counters_sum, op=MPI.SUM, root=0)

    memory: FloatArray = np.array([serial.peak_memory_mb()], dtype=np.float64)
    memory_max: FloatArray = np.empty_like(memory)
    comm.Reduce(memory, memory_max, op=MPI.MAX, root=0)

    hosts: List[str] | None = comm.gather(MPI.Get_processor_name(), root=0)  # once, not hot

    if rank != 0:
        return None, {}

    size: int = comm.Get_size()
    combined: Perf = Perf()
    combined.totals = defaultdict(float, {s: float(v) for s, v in zip(SECTIONS, t_max)})
    combined.calls = perf.calls
    combined.step_times = steps_max.tolist()
    combined.interactions = int(counters_sum[0])
    combined.comm_bytes = int(counters_sum[1])

    t_mean: FloatArray = t_sum / size
    i_forces: int = SECTIONS.index("forces")
    load_imbalance: float = float(t_max[i_forces] / t_mean[i_forces]) if t_mean[i_forces] > 0 else 1.0
    assert hosts is not None
    extra: Dict[str, Value] = {
        "load_imbalance": load_imbalance,
        "n_nodes": len(set(hosts)),
        "hosts": ";".join(sorted(set(hosts))),
        "cpu_model": cpu_model(),
        "decomposition": decomposition_name(args, grid),
        "allgather_variant": "v" if args.decomposition == "allgather" else "",
        "kernel_tile_rows": args.tile_rows,
        "peak_memory_mb": float(memory_max[0]),  # max over ranks (serial: the one process)
        "lsb_jobid": os.environ.get("LSB_JOBID", ""),
        "code_md5": code_fingerprint(),
    }
    for s, mn, mean in zip(SECTIONS, t_min, t_mean):
        if s in ("load", "plotting", "output"):  # rank 0 only, min/mean add nothing
            continue
        extra[f"{s}_time_min_s"] = float(mn)
        extra[f"{s}_time_mean_s"] = float(mean)
    extra["sync_time_s"] = float(t_max[SECTIONS.index("sync")])
    extra["distribute_time_s"] = float(t_max[SECTIONS.index("distribute")])
    return combined, extra


def save_final_state(stars: Stars, path: Path) -> None:
    """Same columns as the serial --save-final, so the two files can be compared directly."""
    pd.DataFrame({
        "x_pc": stars.positions[:, 0], "y_pc": stars.positions[:, 1], "z_pc": stars.positions[:, 2],
        "vx_kms": stars.velocities[:, 0], "vy_kms": stars.velocities[:, 1], "vz_kms": stars.velocities[:, 2],
        "mass_msun": stars.masses,
        "merged_from": stars.members,
        "sptype": stars.sptypes,
        "sptype_guessed": stars.sptype_guessed,
    }).to_csv(path, index=False)


# ---------------------------------------------------------------- command line

def non_negative_int(value: str) -> int:
    n: int = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be 0 or larger")
    return n


def parse_args() -> argparse.Namespace:
    """The serial options (same names, same defaults) plus the MPI-only ones."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="MPI (mpi4py) direct-summation N-body simulation of a Gaia subset. "
                    "Run with mpirun. Same options as SerialImplementation.py.")
    parser.add_argument("--data", type=Path, default=serial.DEFAULT_DATA_PATH,
                        help=f"Subset CSV from GaiaSubsetSelection.py (default: {serial.DEFAULT_DATA_PATH})")
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
                        help="After the run, plot each star's start -> end (rank 0)")
    parser.add_argument("--plot-iter", type=serial.positive_int, default=None, metavar="N",
                        help="Live plot every N iterations (rank 0; costs a gather each time, "
                             "never use it in benchmarks)")
    parser.add_argument("--plots-dir", type=Path, default=serial.DEFAULT_PLOTS_DIR,
                        help=f"Folder the plots are saved in (default: {serial.DEFAULT_PLOTS_DIR})")
    parser.add_argument("--seed", type=int, default=serial.DEFAULT_SEED,
                        help=f"Seed for the random classes of stars without one (default: {serial.DEFAULT_SEED})")
    parser.add_argument("--save-final", type=Path, default=None,
                        help="Save the final state to this CSV (same columns as the serial version)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run in the results file (default: mpi_<mode>_<data>_<P>ranks)")
    parser.add_argument("--results", type=Path, default=serial.DEFAULT_RESULTS_PATH,
                        help=f"CSV that gets one row per run (default: {serial.DEFAULT_RESULTS_PATH})")
    parser.add_argument("--no-results", action="store_true",
                        help="Don't write to the results file")
    # MPI-only
    parser.add_argument("--stop-check-every", type=serial.positive_int, default=1, metavar="K",
                        help="Agree on stopping (time limit / signals) every K steps (default: 1). "
                             "Larger K saves one tiny Allreduce per step but may overshoot the limit")
    parser.add_argument("--decomposition", choices=("allgather", "ring", "2d"), default="allgather",
                        help="allgather: replicate all positions every step (v1). "
                             "ring: pass position blocks around a 1-D periodic Cartesian ring (v2). "
                             "2d: tiles of the pair table on a p_r x p_c Cartesian grid, column Allgatherv "
                             "+ row Allreduce (v3, needs --merge-radius 0). Default: allgather")
    parser.add_argument("--grid-rows", type=non_negative_int, default=0, metavar="R",
                        help="2d only: grid rows p_r (must divide P); 0 = MPI.Compute_dims (default)")
    parser.add_argument("--ring-comm", choices=("overlap", "blocking"), default="overlap",
                        help="ring only. overlap: Isend/Irecv the next block while computing the current "
                             "one. blocking: Sendrecv after each block (default: overlap)")
    parser.add_argument("--tile-rows", type=non_negative_int, default=1, metavar="K",
                        help="Force kernel: K own stars per loop iteration (1 = the v1 kernel, "
                             "0 = automatic, ~8 MB temporaries). Default: 1")
    return parser.parse_args()


# ---------------------------------------------------------------- main

def main() -> None:
    args: argparse.Namespace = parse_args()  # every rank parses the same argv
    # A private copy of the world communicator [W7], passed down as a parameter.
    # The ring runs on a 1-D periodic Cartesian communicator built from it; with
    # reorder=False its ranks are the world ranks, so rank r still owns block r.
    comm: MPI.Comm = MPI.COMM_WORLD.Clone()
    ring: Optional[Ring] = None
    grid: Optional[Grid2D] = None
    if args.decomposition == "ring":
        cart: MPI.Cartcomm = comm.Create_cart([comm.Get_size()], periods=[True], reorder=False)
        comm.Free()
        comm = cart
        ring = Ring(cart, overlap=args.ring_comm == "overlap")
    elif args.decomposition == "2d":
        # Every rank parses the same argv, so every rank stops here together
        if args.merge_radius > 0:
            if comm.Get_rank() == 0:
                print("--decomposition 2d needs --merge-radius 0 (merging is not implemented for 2d)",
                      file=sys.stderr)
            raise SystemExit(2)
        try:
            dims: List[int] = grid_dims(comm.Get_size(), args.grid_rows)
        except ValueError as exc:
            if comm.Get_rank() == 0:
                print(exc, file=sys.stderr)
            raise SystemExit(2)
        # reorder=False: grid rank = world rank, row-major, so rank k is (k // p_c, k % p_c)
        cart = comm.Create_cart(dims, periods=[False, False], reorder=False)
        comm.Free()
        comm = cart
        grid = Grid2D(cart)
    rank: int = comm.Get_rank()
    size: int = comm.Get_size()
    is_root: bool = rank == 0

    run_started: datetime = datetime.now()
    program_start: float = time.perf_counter()
    perf: Perf = Perf()
    mode, time_limit = serial.run_mode(args)

    hosts: List[str] | None = comm.gather(MPI.Get_processor_name(), root=0)
    if is_root:
        assert hosts is not None
        print(f"MPI: {size} ranks on {len(set(hosts))} host(s): {', '.join(sorted(set(hosts)))}")
        print(f"Mode: {mode} (time limit {time_limit:.0f} s, at most {format_thousands(args.steps)} steps)")
        if ring is not None:
            print(f"Decomposition: ring on a 1-D periodic Cartesian communicator ({args.ring_comm}); "
                  f"kernel tile rows: {args.tile_rows or 'auto'}")
        elif grid is not None:
            print(f"Decomposition: 2d on a {grid.pr} x {grid.pc} Cartesian grid (row and column "
                  f"communicators); kernel tile rows: {args.tile_rows or 'auto'}")
        else:
            print(f"Decomposition: allgather (replicated positions); kernel tile rows: {args.tile_rows or 'auto'}")

    state: ParallelState = load_and_distribute(args, comm, perf, grid)
    n_original: int = state.count
    assert state.positions is not None
    start_positions: FloatArray | None = state.positions.copy() if is_root else None  # for --plot

    if is_root:
        counts: IntArray = state.blocks.counts
        if grid is not None:
            cols: IntArray = grid.cols.counts
            print(f"Loaded {format_thousands(n_original)} stars from {args.data}; row blocks of "
                  f"{int(counts.min())}-{int(counts.max())} stars, column blocks of "
                  f"{int(cols.min())}-{int(cols.max())} stars")
        else:
            print(f"Loaded {format_thousands(n_original)} stars from {args.data}; "
                  f"{int(counts.min())}-{int(counts.max())} stars per rank")
        print(f"Total mass: {state.masses.sum():.1f} M_sun")
        print(f"Initial mean distance to centre of mass: "
              f"{serial.mean_distance_to_centre_of_mass(stars_at_start(state)):.3f} pc")
        if args.merge_radius > 0:
            print(f"Merging stars closer than {args.merge_radius} pc")
    if ring is not None or grid is not None:
        state.positions = None  # the ring and 2d keep only the own (row) block from here on

    def gather_state() -> Stars:  # collective
        return gather_grid_state(state, grid, comm, perf) if grid is not None else gather_full_state(state, comm, perf)

    live: serial.LivePlot | None = None
    if args.plot_iter is not None:
        with perf.time("plotting"):
            full: Stars = gather_state()  # collective
            if is_root:
                frames_dir: Path = args.plots_dir / serial.title_to_filename(
                    f"{args.data.stem} mpi live frames")
                live = serial.LivePlot(full, frames_dir)

    deadline: float = program_start + time_limit * (1 - serial.FINISH_RESERVE_FRACTION)
    stop_request: ParallelStopRequest = ParallelStopRequest(rank)
    stop_request.install()
    stop_reason: str = "finished"
    steps_done: int = 0

    comm.Barrier()  # start the step clock together, so no rank times waiting for slower start-ups
    try:
        for i in range(1, args.steps + 1):
            if (i - 1) % args.stop_check_every == 0:
                my_code: int = STOP_INTERRUPTED if stop_request.requested else STOP_NONE
                if is_root:  # only rank 0's clock and window count, so all ranks see one decision
                    if live is not None and not live.is_open():
                        my_code = max(my_code, STOP_WINDOW_CLOSED)
                    expected_step_time: float = perf.step_times[-1] if perf.step_times else 0.0
                    if time.perf_counter() + expected_step_time * args.stop_check_every > deadline:
                        my_code = max(my_code, STOP_TIME_LIMIT)
                code: int = agree_on_stop(comm, my_code, perf)
                if code != STOP_NONE:
                    stop_reason = STOP_REASONS[code]
                    break

            step_start: float = time.perf_counter()
            state, n_merged = parallel_step(state, comm, ring, args, perf, grid)
            perf.step_times.append(time.perf_counter() - step_start)
            steps_done = i
            if n_merged > 0 and is_root:
                print(f"Iteration {format_thousands(i)}: {n_merged} star(s) merged, "
                      f"{format_thousands(state.count)} left")

            if args.plot_iter is not None and i % args.plot_iter == 0:
                with perf.time("plotting"):
                    full = gather_state()  # collective
                    if live is not None and live.is_open():
                        live.update(full, iteration=i, sim_time=i * args.dt)
    finally:
        stop_request.restore()

    simulation_time: float = float(sum(perf.step_times))  # this rank; rank 0 reports the max

    copy_diff: Optional[float] = grid_copy_difference(state, grid, comm) if grid is not None else None
    with perf.time("output"):
        final: Stars = gather_state()                         # collective
    combined, extra = reduce_perf(comm, perf, args, grid)     # collective
    if grid is not None:
        grid.free()
    comm.Free()  # last collective on our communicator
    if copy_diff is not None:
        extra["grid_copy_max_diff"] = copy_diff
        if is_root and copy_diff != 0.0:
            print(f"Warning: the {grid.pc} copies of a row block differ by up to {copy_diff:.3g} "
                  f"(the row Allreduce did not give every rank the same sum)", file=sys.stderr)

    if not is_root:
        return

    # ---- rank 0 only from here: report, save, plot, results row
    assert combined is not None
    simulation_time = float(sum(combined.step_times))  # slowest rank, step by step
    if stop_reason != "finished":
        print(f"Stopped early ({stop_reason}) after {format_thousands(steps_done)} of "
              f"{format_thousands(args.steps)} steps, {time.perf_counter() - program_start:.1f} s "
              f"into a {time_limit:.0f} s limit. Saving what we have.")
    print(f"Final mean distance to centre of mass: {serial.mean_distance_to_centre_of_mass(final):.3f} pc")
    print(f"Stars: {format_thousands(n_original)} at start, {format_thousands(final.count)} at end "
          f"({format_thousands(n_original - final.count)} merged away)")
    print(f"Total momentum at end: {np.round(serial.total_momentum(final), 10)} M_sun·km/s (starts at 0)")
    print(f"Simulated {format_thousands(steps_done)} steps "
          f"({steps_done * args.dt * serial.MYR_PER_TIME_UNIT:.2f} Myr) in {simulation_time:.2f} s "
          f"({simulation_time / max(steps_done, 1) * 1000:.2f} ms/step, slowest rank)")

    if args.save_final is not None:
        with perf.time("output"):
            save_final_state(final, args.save_final)
        print(f"Saved final state to {args.save_final}")

    if live is not None:
        print(f"Saved live plot frames in {live.frames_dir}/")

    if args.plot:
        assert start_positions is not None
        with perf.time("plotting"):
            fig = serial.plot_start_to_end(start_positions, final)
            title: str = (f"{args.data.stem} — mpi {size} ranks — {format_thousands(n_original)} stars — "
                          f"Start to end after {format_thousands(steps_done)} iterations · "
                          f"t = {steps_done * args.dt * serial.MYR_PER_TIME_UNIT:.2f} Myr")
            if final.count < n_original:
                title += f" — {format_thousands(final.count)} after merging"
            saved_plot: Path = serial.save_figure(fig, args.plots_dir, title)
        print(f"Saved plot: {saved_plot}")

    total_time: float = time.perf_counter() - program_start
    # The final save and plot happen after reduce_perf, so add rank 0's own time for them
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
          f"load imbalance (forces max/mean) {float(row['load_imbalance']):.3f}")
    if not args.no_results:
        serial.append_results(args.results, row)
        print(f"Added run '{row['run_name']}' to {args.results}")

    if (args.plot or live is not None) and matplotlib.get_backend().lower() != "agg":
        plt.ioff()
        plt.show()


def stars_at_start(state: ParallelState) -> Stars:
    """A Stars view of the replicated arrays, valid only while positions are current
    (at start-up, before anything has moved). Velocities are not needed for it."""
    assert state.positions is not None
    return Stars(positions=state.positions, velocities=np.zeros_like(state.positions),
                 masses=state.masses, members=state.members, owner=state.owner,
                 sptypes=state.sptypes, sptype_guessed=state.sptype_guessed)


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
