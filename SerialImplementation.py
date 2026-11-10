"""Serial direct-summation N-body simulation of a Gaia star subset.

Loads a subset CSV made by GaiaSubsetSelection.py (which already contains the
Cartesian positions x_pc, y_pc, z_pc) and simulates every star in it.

Stars that come closer than --merge-radius merge into one star at their centre
of mass, with the combined mass and momentum. A merged star keeps the spectral
class of its most massive member, preferring members with a real (not guessed)
class.

In the plots, stars are coloured by spectral class (SpType-ELS: O B A F G K M)
and sized by mass, on a dark background. Stars without a class get a random
F/G/K/M class, weighted by how common each type is, so most become K and M
dwarfs (seeded with --seed, so every run gives the same draw).

Units (chosen so the numbers stay reasonable for stars):
    position  : parsec (pc)
    velocity  : km/s
    mass      : solar masses (M_sun), from the Mass-Flame column
    time      : pc / (km/s)  ≈ 0.978 million years

Usage examples:
    python SerialImplementation.py                      # normal run: at most 5 minutes, no plotting
    python SerialImplementation.py --quick              # quick run: at most 15 seconds
    python SerialImplementation.py --plot               # start → end plot after the run
    python SerialImplementation.py --plot-iter 100      # live plot every 100 iterations
    python SerialImplementation.py --merge-radius 0.05  # merge stars closer than 0.05 pc
    python SerialImplementation.py --merge-radius 0     # no merging
    python SerialImplementation.py --data gaia_subsets/gaia_random_5000.csv --steps 2000
    python SerialImplementation.py --save-final final_state.csv
    python SerialImplementation.py --run-name baseline_n1000 --results results.csv

Every run appends one row of performance results to results.csv (see
build_results_row for the columns); use --no-results to skip that.

Runs stop at whichever comes first: --steps, or the time limit (--quick: 15 s,
normal: 5 min, or --time-limit SECONDS). Ctrl+C or SIGTERM (e.g. a cluster
job being cancelled) also stops after the current step. In every case the
results row, --save-final and the plots are still saved, marked with
completed=False and the stop_reason.

Plots are saved as PNGs in the Plots folder (change with --plots-dir), named
after what their title would have been; the plots themselves have no titles.
"""

import argparse
import os
import platform
import re
import signal
import socket
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, Dict, Iterator, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]  # (N, 3) positions/velocities or (N,) masses
IntArray = NDArray[np.int64]
StrArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]
Pair = Tuple[int, int]

# Gravitational constant in pc · (km/s)² / M_sun
G: float = 4.30091e-3

# 1 time unit = 1 pc / (km/s) in millions of years
MYR_PER_TIME_UNIT: float = 0.9778

DEFAULT_DATA_PATH: Path = Path("gaia_subsets/gaia_nearest_1000.csv")
DEFAULT_MASS: float = 1.0  # M_sun, used where Mass-Flame is missing
DEFAULT_PLOTS_DIR: Path = Path("Plots")

# Approximate colour of each spectral class as the eye would see it
# (blackbody colours, after Mitchell Charity's "What color are the stars?")
SPECTRAL_CLASSES: str = "OBAFGKM"
SPECTRAL_COLOURS: Dict[str, str] = {
    "O": "#9bb0ff",  # blue
    "B": "#aabfff",  # blue-white
    "A": "#cad7ff",  # white-blue
    "F": "#f8f7ff",  # white
    "G": "#fff4ea",  # yellow-white
    "K": "#ffd2a1",  # orange
    "M": "#ffcc6f",  # orange-red
}
UNKNOWN_CLASS: str = "?"
UNKNOWN_COLOUR: str = "#808080"

# Share of main-sequence stars per class (LeDrew 2001, "The Real Starry Sky",
# as listed on Wikipedia's "Stellar classification" page). Stars without a class
# are drawn from F/G/K/M with these weights, i.e. after normalising:
# F ≈ 3 %, G ≈ 8 %, K ≈ 12 %, M ≈ 77 % (red dwarfs dominate).
GUESS_CLASS_WEIGHTS: Dict[str, float] = {"F": 3.0, "G": 7.6, "K": 12.1, "M": 76.45}
DEFAULT_SEED: int = 42
DEFAULT_RESULTS_PATH: Path = Path("results.csv")

# Time limits (seconds of wall time for the whole program, plotting included)
QUICK_TIME_LIMIT_S: float = 15.0
NORMAL_TIME_LIMIT_S: float = 300.0
# Stop stepping this share of the limit early, to leave time to save results and plots
FINISH_RESERVE_FRACTION: float = 0.05

# Performance model (see record_results for how these are used)
FLOPS_PER_INTERACTION: int = 20     # common N-body convention: 3 sub, 3 mul, 2 add (r²),
                                    # 1 add (softening), ~4 for r^-1.5, 1 mul (m·r^-3), 3 mul + 3 add (sum)
BYTES_PER_INTERACTION: int = 32     # minimum data read per pair: position of j (3 × 8 B) + mass of j (8 B)


# ---------------------------------------------------------------- star data

@dataclass
class Stars:
    """All stars currently in the simulation.

    positions, velocities and masses shrink when stars merge. `owner` always has
    one entry per *original* star and says which current star it is now part of,
    so start positions can still be matched to end positions after merges.
    """
    positions: FloatArray   # (N, 3) pc
    velocities: FloatArray  # (N, 3) km/s
    masses: FloatArray      # (N,)   M_sun
    members: IntArray       # (N,)   how many original stars each star is made of
    owner: IntArray         # (N_original,) index of the current star each original star belongs to
    sptypes: StrArray       # (N,)   spectral class O/B/A/F/G/K/M
    sptype_guessed: BoolArray  # (N,) True if the class was drawn at random (no SpType-ELS)

    @property
    def count(self) -> int:
        return len(self.masses)


def load_subset(file_path: Path, rng: np.random.Generator) -> Stars:
    """Load a subset CSV; every star starts at rest.

    `rng` is used to draw a class for stars that have none.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}. Create it with "
                                f"`python GaiaSubsetSelection.py --save` or pass --data <path>.")

    df: pd.DataFrame = pd.read_csv(file_path)
    # pandas returns a column-major (Fortran-ordered) array here. Make it row-major
    # (C-contiguous) so that x, y, z of one star sit next to each other in memory:
    # MPI sends raw memory, and the summation order (hence the last bits) must match the MPI version.
    positions: FloatArray = np.ascontiguousarray(df[["x_pc", "y_pc", "z_pc"]].to_numpy(dtype=np.float64))

    if "Mass-Flame" in df.columns:
        masses: FloatArray = df["Mass-Flame"].to_numpy(dtype=np.float64, copy=True)
        invalid: NDArray[np.bool_] = ~np.isfinite(masses) | (masses <= 0)
        n_missing: int = int(invalid.sum())
        if n_missing > 0:
            print(f"{n_missing:,} stars have no valid Mass-Flame, using {DEFAULT_MASS} M_sun for them")
            masses[invalid] = DEFAULT_MASS
    else:
        print(f"No Mass-Flame column, using {DEFAULT_MASS} M_sun for every star")
        masses = np.full(len(df), DEFAULT_MASS, dtype=np.float64)

    n: int = len(masses)
    if "SpType-ELS" in df.columns:
        sptypes: StrArray = parse_spectral_classes(df["SpType-ELS"])
    else:
        print("No SpType-ELS column, every star gets a random class")
        sptypes = np.full(n, UNKNOWN_CLASS, dtype="<U1")

    sptype_guessed: BoolArray = sptypes == UNKNOWN_CLASS
    n_guessed: int = int(sptype_guessed.sum())
    if n_guessed > 0:
        sptypes[sptype_guessed] = guess_spectral_classes(n_guessed, rng)
        drawn: str = ", ".join(f"{cls} {int(np.sum(sptypes[sptype_guessed] == cls))}"
                               for cls in GUESS_CLASS_WEIGHTS)
        print(f"{n_guessed:,} stars have no O/B/A/F/G/K/M class; drew one at random: {drawn}")

    return Stars(
        positions=positions,
        velocities=np.zeros_like(positions),  # TODO: initial velocities from pmRA, pmDE, RV
        masses=masses,
        members=np.ones(n, dtype=np.int64),
        owner=np.arange(n, dtype=np.int64),
        sptypes=sptypes,
        sptype_guessed=sptype_guessed,
    )


def guess_spectral_classes(n: int, rng: np.random.Generator) -> StrArray:
    """Draw n classes from F/G/K/M, weighted by how common each type is (mostly M dwarfs)."""
    classes: List[str] = list(GUESS_CLASS_WEIGHTS)
    weights: FloatArray = np.array(list(GUESS_CLASS_WEIGHTS.values()), dtype=np.float64)
    return rng.choice(classes, size=n, p=weights / weights.sum()).astype("<U1")


def parse_spectral_classes(column: pd.Series) -> StrArray:
    """'O      ' -> 'O', 'g' -> 'G'; anything that doesn't start with O/B/A/F/G/K/M -> '?'."""
    first_letters: pd.Series = column.fillna("").astype(str).str.strip().str[:1].str.upper()
    return np.where(first_letters.isin(list(SPECTRAL_CLASSES)), first_letters, UNKNOWN_CLASS).astype("<U1")


# ---------------------------------------------------------------- performance

@dataclass
class Perf:
    """Collects timings and counters for one run.

    Sections timed (seconds, summed over the whole run):
        load           reading the CSV and setting up the stars
        forces         compute_accelerations (incl. finding close pairs)
        merge          merge_stars
        integrate      kick + drift (velocity and position updates)
        communication  MPI data exchange (always 0 in the serial version)
        plotting       live plot updates, final plot and saving the PNGs
        output         writing --save-final
    """
    totals: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    step_times: List[float] = field(default_factory=list)  # wall time of every step (s), plotting excluded
    interactions: int = 0     # pair interactions computed: Σ N·(N−1) over every force evaluation
    comm_bytes: int = 0       # bytes sent between ranks (MPI only)

    @contextmanager
    def time(self, section: str) -> Iterator[None]:
        start: float = time.perf_counter()
        try:
            yield
        finally:
            self.totals[section] += time.perf_counter() - start
            self.calls[section] += 1


def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (NaN where not available, e.g. Windows)."""
    try:
        import resource
        peak: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024 if sys.platform != "darwin" else peak / 1024 ** 2  # Linux: KB, macOS: bytes
    except (ImportError, OSError):
        return float("nan")


# ---------------------------------------------------------------- physics

def compute_accelerations(positions: FloatArray, masses: FloatArray, softening: float,
                          merge_radius: float, i_start: int = 0,
                          i_stop: int | None = None) -> Tuple[FloatArray, List[Pair]]:
    """Gravitational acceleration on stars i_start..i_stop-1 from every star (direct summation).

    Each star i is handled independently (a loop over i, vectorised over j),
    which is the same split an MPI version can use: give each rank a block of i's.
    The serial code uses the default range (all stars). ParallelImplementation.py
    passes each rank's own block, so both versions run exactly this kernel.

    `positions` and `masses` must always hold *all* N stars (every j), and `i`
    stays the global star index inside the loop: the self-force exclusion
    (dist_sq[i]) and the pair search (j > i) both depend on it.

    `softening` (pc) is added to every distance so that very close passes don't
    give an almost infinite kick.

    Returns the accelerations of the block, shape (i_stop - i_start, 3), and
    every pair (i, j), i < j, with i in the block, that is closer than
    `merge_radius`. This reuses the distances already computed for the forces,
    so finding merges costs (almost) nothing extra. Because only j > i is
    searched, each pair is found exactly once, by whoever owns the smaller index.
    """
    n: int = len(masses)
    if i_stop is None:
        i_stop = n
    accelerations: FloatArray = np.zeros((i_stop - i_start, 3), dtype=np.float64)
    softening_sq: float = softening ** 2
    merge_radius_sq: float = merge_radius ** 2
    close_pairs: List[Pair] = []

    for i in range(i_start, i_stop):
        dis_vectors: FloatArray = positions - positions[i]          # from star i to every star j
        raw_dist_sq: FloatArray = np.sum(dis_vectors ** 2, axis=1)

        if merge_radius > 0:
            # only look at j > i so each pair is found once
            close_j: IntArray = np.nonzero(raw_dist_sq[i + 1:] < merge_radius_sq)[0] + i + 1
            close_pairs.extend((i, int(j)) for j in close_j)

        dist_sq: FloatArray = raw_dist_sq + softening_sq
        dist_sq[i] = np.inf                                           # no force from itself
        inv_dist_cubed: FloatArray = dist_sq ** -1.5
        # a_i = G · Σ_j m_j · (r_j − r_i) / |r_j − r_i|³
        accelerations[i - i_start] = G * np.sum((masses * inv_dist_cubed)[:, np.newaxis] * dis_vectors, axis=0)

    return accelerations, close_pairs


def group_close_stars(n: int, pairs: List[Pair]) -> IntArray:
    """Give every star a group label; stars linked by close pairs share a label.

    Uses union-find, so chains (A close to B, B close to C) end up in one group.
    Labels are 0..n_groups-1 and keep the original order of the stars.
    """
    parent: IntArray = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path halving keeps the trees flat
            a = int(parent[a])
        return a

    for a, b in pairs:
        root_a: int = find(a)
        root_b: int = find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    roots: IntArray = np.array([find(k) for k in range(n)], dtype=np.int64)
    labels: IntArray = np.unique(roots, return_inverse=True)[1].astype(np.int64)
    return labels


def merge_stars(stars: Stars, pairs: List[Pair]) -> Stars:
    """Merge every group of close stars into one star.

    The new star has the total mass, sits at the group's centre of mass and
    moves with velocity = total momentum / total mass, so mass and momentum
    are conserved exactly.
    """
    labels: IntArray = group_close_stars(stars.count, pairs)
    n_groups: int = int(labels.max()) + 1

    masses: FloatArray = np.bincount(labels, weights=stars.masses, minlength=n_groups)
    positions: FloatArray = np.empty((n_groups, 3), dtype=np.float64)
    velocities: FloatArray = np.empty((n_groups, 3), dtype=np.float64)
    for axis in range(3):
        mass_weighted_pos: FloatArray = np.bincount(
            labels, weights=stars.masses * stars.positions[:, axis], minlength=n_groups)
        momentum: FloatArray = np.bincount(
            labels, weights=stars.masses * stars.velocities[:, axis], minlength=n_groups)
        positions[:, axis] = mass_weighted_pos / masses
        velocities[:, axis] = momentum / masses

    members: IntArray = np.bincount(labels, weights=stars.members, minlength=n_groups).astype(np.int64)

    # A merged star keeps the spectral class of its most massive member, preferring
    # members whose class is real over ones that were guessed. Sort by group, then
    # real/guessed, then mass: the last star of each group is the one to copy.
    real: BoolArray = ~stars.sptype_guessed
    order: IntArray = np.lexsort((stars.masses, real, labels))
    sorted_labels: IntArray = labels[order]
    last_of_group: IntArray = np.append(np.nonzero(np.diff(sorted_labels))[0], len(order) - 1)
    chosen: IntArray = order[last_of_group]

    return Stars(positions=positions, velocities=velocities, masses=masses,
                 members=members, owner=labels[stars.owner],
                 sptypes=stars.sptypes[chosen], sptype_guessed=stars.sptype_guessed[chosen])


def timed_accelerations(stars: Stars, softening: float, merge_radius: float,
                        perf: Perf) -> Tuple[FloatArray, List[Pair]]:
    with perf.time("forces"):
        result: Tuple[FloatArray, List[Pair]] = compute_accelerations(
            stars.positions, stars.masses, softening, merge_radius)
    perf.interactions += stars.count * (stars.count - 1)
    return result


def step(stars: Stars, dt: float, softening: float, merge_radius: float, perf: Perf) -> Tuple[Stars, int]:
    """Advance one time step. Returns the updated stars and how many stars merged away.

    Order: find close pairs → merge them → compute accelerations → kick → drift.
    All accelerations are computed from the current positions *before* anything
    moves, so the order in which stars are processed doesn't matter.
    """
    n_before: int = stars.count
    accelerations, close_pairs = timed_accelerations(stars, softening, merge_radius, perf)

    # Merging moves stars to new centres of mass, which can bring them within
    # merge_radius of yet another star, so repeat until nothing is close.
    while close_pairs:
        with perf.time("merge"):
            stars = merge_stars(stars, close_pairs)
        accelerations, close_pairs = timed_accelerations(stars, softening, merge_radius, perf)

    with perf.time("integrate"):
        stars.velocities += accelerations * dt
        stars.positions += stars.velocities * dt
    return stars, n_before - stars.count


def mean_distance_to_centre_of_mass(stars: Stars) -> float:
    centre: FloatArray = np.average(stars.positions, axis=0, weights=stars.masses)
    return float(np.mean(np.linalg.norm(stars.positions - centre, axis=1)))


def total_momentum(stars: Stars) -> FloatArray:
    return np.sum(stars.masses[:, np.newaxis] * stars.velocities, axis=0)


# ---------------------------------------------------------------- plotting

def format_thousands(n: int, sep: str = ".") -> str:
    return f"{n:,}".replace(",", sep)


def title_to_filename(title: str) -> str:
    """Turn a would-be plot title into a safe file name, e.g.
    'gaia_nearest_1000 — Iteration 1.000 · t = 9.78 Myr' -> 'gaia_nearest_1000_Iteration_1_000_t_9_78_Myr'."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", title).strip("_")


def save_figure(fig: Figure, folder: Path, title: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path: Path = folder / f"{title_to_filename(title)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# Dark "space" background, so the pale star colours stand out
SPACE_BLACK: str = "#05060a"
PANE_COLOUR: Tuple[float, float, float, float] = (0.06, 0.07, 0.09, 1.0)
GRID_COLOUR: str = "#2a2e36"
TEXT_COLOUR: str = "#b8bec8"
TRAIL_COLOUR: str = "#5a606b"


def new_3d_axes() -> Tuple[Figure, Axes3D]:
    fig: Figure = plt.figure(figsize=(8, 7), facecolor=SPACE_BLACK)
    ax: Axes3D = fig.add_subplot(projection="3d", facecolor=SPACE_BLACK)
    for axis, label in ((ax.xaxis, "x [pc]"), (ax.yaxis, "y [pc]"), (ax.zaxis, "z [pc]")):
        axis.set_label_text(label, color=TEXT_COLOUR)
        axis.set_pane_color(PANE_COLOUR)
        axis.line.set_color(TEXT_COLOUR)
        axis._axinfo["grid"]["color"] = GRID_COLOUR  # only way to recolour 3D grid lines
    ax.tick_params(colors=TEXT_COLOUR)
    return fig, ax


def spectral_colours(sptypes: StrArray) -> List[str]:
    return [SPECTRAL_COLOURS.get(str(t), UNKNOWN_COLOUR) for t in sptypes]


def add_class_legend(ax: Axes3D, stars: Stars, extra_handles: List[Line2D] | None = None) -> None:
    """Legend with one entry per spectral class present, e.g. 'K  (12, 5 guessed)'."""
    handles: List[Line2D] = list(extra_handles or [])
    for cls in SPECTRAL_CLASSES + UNKNOWN_CLASS:
        is_cls: BoolArray = stars.sptypes == cls
        count: int = int(is_cls.sum())
        if count == 0:
            continue
        n_guessed: int = int(np.sum(is_cls & stars.sptype_guessed))
        colour: str = SPECTRAL_COLOURS.get(cls, UNKNOWN_COLOUR)
        label: str = "unknown" if cls == UNKNOWN_CLASS else cls
        counts: str = format_thousands(count) + (f", {format_thousands(n_guessed)} guessed" if n_guessed else "")
        handles.append(Line2D([], [], linestyle="", marker="o", markersize=6, color=colour,
                              label=f"{label}  ({counts})"))
    ax.legend(handles=handles, loc="upper left", facecolor=SPACE_BLACK, edgecolor=GRID_COLOUR,
              labelcolor=TEXT_COLOUR, title="Class", title_fontsize="small")
    ax.get_legend().get_title().set_color(TEXT_COLOUR)


def set_equal_limits(ax: Axes3D, points: FloatArray) -> None:
    """Fit the axes around `points` with the same scale on every axis."""
    lo: FloatArray = points.min(axis=0)
    hi: FloatArray = points.max(axis=0)
    centre: FloatArray = (lo + hi) / 2
    half: float = max(float((hi - lo).max()) / 2 * 1.1, 1e-9)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)


def marker_sizes(masses: FloatArray, reference_mass: float) -> FloatArray:
    """Marker area grows with √mass, so merged stars are bigger without filling the plot."""
    return 2 + 6 * np.sqrt(masses / reference_mass)


# -------- live plot (--plot-iter N)

class LivePlot:
    """Scatter of all stars, redrawn (and saved) while the simulation runs.

    Every frame is saved as a PNG in `frames_dir`. The would-be title is shown
    in the window's title bar instead of on the plot.
    """

    def __init__(self, stars: Stars, frames_dir: Path) -> None:
        plt.ion()  # interactive mode: the window updates without blocking
        self.frames_dir: Path = frames_dir
        # Size scale is fixed from the start, so merged stars stand out
        self.reference_mass: float = float(stars.masses.max())
        self.fig, self.ax = new_3d_axes()
        self.scatter: Path3DCollection = self.ax.scatter(
            stars.positions[:, 0], stars.positions[:, 1], stars.positions[:, 2],
            s=marker_sizes(stars.masses, self.reference_mass), c=spectral_colours(stars.sptypes),
            alpha=0.9, linewidths=0, depthshade=False)
        # Fixed limits from the starting positions, so movement is visible
        # instead of the axes chasing the stars
        set_equal_limits(self.ax, stars.positions)
        self.update(stars, iteration=0, sim_time=0.0)

    def update(self, stars: Stars, iteration: int, sim_time: float) -> None:
        # The number of stars can shrink, so positions, sizes and colours are all replaced
        self.scatter._offsets3d = (stars.positions[:, 0], stars.positions[:, 1], stars.positions[:, 2])
        self.scatter.set_sizes(marker_sizes(stars.masses, self.reference_mass))
        self.scatter.set_facecolor(spectral_colours(stars.sptypes))
        add_class_legend(self.ax, stars)
        title: str = (f"Iteration {format_thousands(iteration)} · "
                      f"t = {sim_time * MYR_PER_TIME_UNIT:.2f} Myr · "
                      f"{format_thousands(stars.count)} stars")
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(title)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # lets the GUI process the redraw
        save_figure(self.fig, self.frames_dir, title)

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)


# -------- final plot (--plot)

def plot_start_to_end(start_positions: FloatArray, stars: Stars) -> Figure:
    """One straight line per original star, from where it started to where it ended.

    A star that merged ends at the position of the star it merged into, so all
    lines of a merged group meet in one point.
    """
    fig, ax = new_3d_axes()
    end_positions: FloatArray = stars.positions[stars.owner]  # (N_original, 3)

    # (N_original, 2, 3): for each original star a segment [start, end]
    segments: NDArray[np.float64] = np.stack([start_positions, end_positions], axis=1)
    lines: Line3DCollection = Line3DCollection(segments, colors=TRAIL_COLOUR, linewidths=0.5, alpha=0.6)
    ax.add_collection3d(lines)

    # Start points: small and grey; end points: coloured by spectral class, sized by mass
    ax.scatter(start_positions[:, 0], start_positions[:, 1], start_positions[:, 2],
               s=3, c=TRAIL_COLOUR, linewidths=0, depthshade=False)
    ax.scatter(stars.positions[:, 0], stars.positions[:, 1], stars.positions[:, 2],
               s=marker_sizes(stars.masses, float(np.median(stars.masses))) * 2,
               c=spectral_colours(stars.sptypes), linewidths=0, depthshade=False)

    start_handle: Line2D = Line2D([], [], linestyle="-", marker="o", markersize=3,
                                  color=TRAIL_COLOUR, label="start → end")
    add_class_legend(ax, stars, extra_handles=[start_handle])
    # Line collections don't autoscale, so fit the axes to start and end points
    set_equal_limits(ax, np.vstack([start_positions, stars.positions]))
    return fig


# ---------------------------------------------------------------- stopping early

class StopRequest:
    """Turns Ctrl+C (SIGINT) and SIGTERM into a polite "stop after this step".

    SIGTERM is what cluster schedulers (e.g. SLURM) send before killing a job,
    so a run that is cut off still saves its results. A second Ctrl+C stops
    immediately (the results are still written).
    """

    def __init__(self) -> None:
        self.requested: bool = False
        self._previous: Dict[int, Any] = {}

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.signal(sig, self._handle)

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print(f"\nReceived {signal.Signals(signum).name}: stopping after this step and saving "
              f"(press Ctrl+C again to stop at once)")


def run_mode(args: argparse.Namespace) -> Tuple[str, float]:
    """('quick' | 'normal' | 'custom', time limit in seconds)."""
    if args.time_limit is not None:
        return "custom", args.time_limit
    if args.quick:
        return "quick", QUICK_TIME_LIMIT_S
    return "normal", NORMAL_TIME_LIMIT_S


# ---------------------------------------------------------------- results

Value = Union[str, int, float, bool]


def default_run_name(args: argparse.Namespace, implementation: str, n_ranks: int) -> str:
    return f"{implementation}_{run_mode(args)[0]}_{args.data.stem}_{n_ranks}ranks"


def build_results_row(*, args: argparse.Namespace, perf: Perf, run_started: datetime,
                      total_time: float, simulation_time: float, iterations: int,
                      n_stars_start: int, stars: Stars, stop_reason: str,
                      implementation: str = "serial", n_ranks: int = 1, n_threads: int = 1,
                      load_imbalance: float = 1.0) -> Dict[str, Value]:
    """One row for results.csv.

    The MPI version can call this too: pass implementation="mpi", n_ranks, and
    load_imbalance (max / mean compute time over ranks), and fill perf.comm_bytes
    and perf.totals["communication"].

    Derived metrics:
        interactions_per_s   pair interactions / forces time
        est_gflops           interactions × FLOPS_PER_INTERACTION / forces time
        model_bandwidth_GBps interactions × BYTES_PER_INTERACTION / forces time — the
                             *minimum* memory traffic the algorithm needs; NumPy's
                             temporary arrays move more than this in reality
        comm_bandwidth_GBps  comm_bytes / communication time (MPI only)
    """
    t: Dict[str, float] = perf.totals
    forces: float = t["forces"]
    comm: float = t["communication"]
    steps_ms: FloatArray = np.array(perf.step_times, dtype=np.float64) * 1000
    model_bytes: int = perf.interactions * BYTES_PER_INTERACTION

    def per_second(amount: float, seconds: float) -> float:
        return amount / seconds if seconds > 0 else float("nan")

    return {
        # --- run
        "timestamp": run_started.isoformat(timespec="seconds"),
        "run_name": args.run_name or default_run_name(args, implementation, n_ranks),
        "implementation": implementation,
        "n_ranks": n_ranks,
        "n_threads": n_threads,
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count() or 0,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        # --- problem
        "data_file": str(args.data),
        "n_stars_start": n_stars_start,
        "n_stars_end": stars.count,
        "n_merged": n_stars_start - stars.count,
        "mode": run_mode(args)[0],
        "time_limit_s": run_mode(args)[1],
        "completed": iterations == args.steps,
        "stop_reason": stop_reason,     # finished | time_limit | interrupted | window_closed | error: ...
        "steps_requested": args.steps,
        "iterations": iterations,
        "dt": args.dt,
        "softening_pc": args.softening,
        "merge_radius_pc": args.merge_radius,
        "seed": args.seed,
        "simulated_myr": iterations * args.dt * MYR_PER_TIME_UNIT,
        "plot": args.plot,
        "plot_iter": args.plot_iter or 0,
        # --- time per section (s)
        "total_time_s": total_time,
        "load_time_s": t["load"],
        "simulation_time_s": simulation_time,
        "forces_time_s": forces,
        "merge_time_s": t["merge"],
        "integrate_time_s": t["integrate"],
        "communication_time_s": comm,
        "other_sim_time_s": simulation_time - forces - t["merge"] - t["integrate"] - comm,
        "plotting_time_s": t["plotting"],
        "output_time_s": t["output"],
        "forces_calls": perf.calls["forces"],
        "merge_calls": perf.calls["merge"],
        # --- per step (ms, plotting excluded)
        "step_time_mean_ms": float(steps_ms.mean()) if steps_ms.size else float("nan"),
        "step_time_std_ms": float(steps_ms.std()) if steps_ms.size else float("nan"),
        "step_time_min_ms": float(steps_ms.min()) if steps_ms.size else float("nan"),
        "step_time_max_ms": float(steps_ms.max()) if steps_ms.size else float("nan"),
        # --- throughput / bandwidth
        "pair_interactions": perf.interactions,
        "interactions_per_s": per_second(perf.interactions, forces),
        "est_gflops": per_second(perf.interactions * FLOPS_PER_INTERACTION, forces) / 1e9,
        "model_memory_bytes": model_bytes,
        "model_bandwidth_GBps": per_second(model_bytes, forces) / 1e9,
        # --- parallel (filled by the MPI version)
        "comm_bytes": perf.comm_bytes,
        "comm_bandwidth_GBps": per_second(perf.comm_bytes, comm) / 1e9,
        "load_imbalance": load_imbalance,
        # --- resources
        "peak_memory_mb": peak_memory_mb(),
        # --- validation (should match between serial and MPI for the same inputs)
        "final_mean_dist_com_pc": mean_distance_to_centre_of_mass(stars),
        "final_momentum_norm": float(np.linalg.norm(total_momentum(stars))),
        "final_position_checksum": float(np.sum(stars.positions)),
    }


@contextmanager
def results_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on `<results>.lock` while the results file is updated.

    Several batch jobs can finish at the same moment and write to the same
    results.csv. Without a lock, two jobs can both read the file, each add their
    row, and one row is lost. POSIX locks (fcntl.lockf) also work on the
    cluster's network file systems. On Windows (no fcntl) there is no lock,
    which is fine for one run at a time.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock_file:
        fcntl.lockf(lock_file, fcntl.LOCK_EX)  # waits until no other job holds the lock
        try:
            yield
        finally:
            fcntl.lockf(lock_file, fcntl.LOCK_UN)


def append_results(path: Path, row: Dict[str, Value]) -> None:
    """Append one row to the results CSV, creating it (with header) if needed.

    Serial and MPI runs share one file. Columns a run doesn't have (e.g. the
    MPI-only ones for a serial run) are left empty. If a run brings new columns,
    the file is rewritten with the merged columns, so adding a metric later
    never breaks the file. Safe when several jobs write at the same time.
    """
    new_row: pd.DataFrame = pd.DataFrame([row])
    with results_lock(path):
        if path.exists() and path.stat().st_size > 0:
            columns: List[str] = list(pd.read_csv(path, nrows=0).columns)  # header only
            if set(new_row.columns) <= set(columns):
                # Same or fewer columns: append in the file's column order
                new_row.reindex(columns=columns).to_csv(path, mode="a", header=False, index=False)
                return
            new_row = pd.concat([pd.read_csv(path), new_row], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_row.to_csv(path, index=False)


# ---------------------------------------------------------------- main

def positive_int(value: str) -> int:
    n: int = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or larger")
    return n


def non_negative_float(value: str) -> float:
    x: float = float(value)
    if x < 0:
        raise argparse.ArgumentTypeError("must be 0 or larger")
    return x


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Serial direct-summation N-body simulation of a Gaia subset. "
                    "No plotting by default.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help=f"Subset CSV from GaiaSubsetSelection.py (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--quick", action="store_true",
                        help=f"Quick run: stop after at most {QUICK_TIME_LIMIT_S:.0f} s "
                             f"(default without it: {NORMAL_TIME_LIMIT_S / 60:.0f} min)")
    parser.add_argument("--time-limit", type=non_negative_float, default=None, metavar="SECONDS",
                        help="Custom time limit, overrides --quick / the normal limit")
    parser.add_argument("--steps", type=positive_int, default=5_000,
                        help="Maximum number of time steps; the run also stops at the time limit "
                             "(default: 5000)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step in pc/(km/s) ≈ 0.978 Myr (default: 0.01 ≈ 9,800 years)")
    parser.add_argument("--softening", type=non_negative_float, default=0.01,
                        help="Softening length in pc (default: 0.01)")
    parser.add_argument("--merge-radius", type=non_negative_float, default=0.01,
                        help="Stars closer than this (pc) merge into one; 0 turns merging off "
                             "(default: 0.01 ≈ 2,000 AU)")
    parser.add_argument("--plot", action="store_true",
                        help="After the run, plot a line from each star's start to its end position")
    parser.add_argument("--plot-iter", type=positive_int, default=None, metavar="N",
                        help="Show a live plot while running, redrawn every N iterations")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR,
                        help=f"Folder the plots are saved in (default: {DEFAULT_PLOTS_DIR})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Seed for the random classes of stars without one (default: {DEFAULT_SEED})")
    parser.add_argument("--save-final", type=Path, default=None,
                        help="Save final positions and velocities to this CSV, e.g. to compare with the MPI version")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run in the results file "
                             "(default: serial_<mode>_<data>_1ranks)")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH,
                        help=f"CSV that gets one row of performance results per run (default: {DEFAULT_RESULTS_PATH})")
    parser.add_argument("--no-results", action="store_true",
                        help="Don't write to the results file (e.g. for quick tests)")
    return parser.parse_args()


def main() -> None:
    # TODO: Run on one star - DONE
    # TODO: Run on two stars - DONE
    # TODO: Find distance between two stars - DONE
    # TODO: Create movement simulation - DONE
    # TODO: Plot movement - DONE
    # TODO: Run on subset - DONE
    # TODO: Merge stars that get too close - DONE
    # TODO: Inertia, is that a thing? (initial velocities from pmRA, pmDE, RV)
    args: argparse.Namespace = parse_args()
    run_started: datetime = datetime.now()
    program_start: float = time.perf_counter()
    perf: Perf = Perf()
    mode, time_limit = run_mode(args)
    print(f"Mode: {mode} (time limit {time_limit:.0f} s, at most {format_thousands(args.steps)} steps)")

    with perf.time("load"):
        stars: Stars = load_subset(args.data, np.random.default_rng(args.seed))
    start_positions: FloatArray = stars.positions.copy()
    n_original: int = stars.count

    print(f"Loaded {format_thousands(n_original)} stars from {args.data}")
    print(f"Total mass: {stars.masses.sum():.1f} M_sun")
    print(f"Initial mean distance to centre of mass: {mean_distance_to_centre_of_mass(stars):.3f} pc")
    if args.merge_radius > 0:
        print(f"Merging stars closer than {args.merge_radius} pc")

    live: LivePlot | None = None
    if args.plot_iter is not None:
        with perf.time("plotting"):
            # One sub-folder per run so the frames of different datasets don't mix
            frames_dir: Path = args.plots_dir / title_to_filename(f"{args.data.stem} live frames")
            live = LivePlot(stars, frames_dir)

    # Stop starting new steps once the next one would end past the deadline;
    # the reserve leaves time to save results and plots within the limit.
    deadline: float = program_start + time_limit * (1 - FINISH_RESERVE_FRACTION)
    stop_request: StopRequest = StopRequest()
    stop_request.install()
    stop_reason: str = "finished"
    error: Exception | None = None

    steps_done: int = 0
    try:
        for i in range(1, args.steps + 1):
            if stop_request.requested:
                stop_reason = "interrupted"
                break
            expected_step_time: float = perf.step_times[-1] if perf.step_times else 0.0
            if time.perf_counter() + expected_step_time > deadline:
                stop_reason = "time_limit"
                break

            step_start: float = time.perf_counter()
            stars, n_merged = step(stars, args.dt, args.softening, args.merge_radius, perf)
            perf.step_times.append(time.perf_counter() - step_start)
            steps_done = i
            if n_merged > 0:
                print(f"Iteration {format_thousands(i)}: {n_merged} star(s) merged, "
                      f"{format_thousands(stars.count)} left")

            if live is not None and i % args.plot_iter == 0:
                if not live.is_open():  # closing the window stops the run
                    stop_reason = "window_closed"
                    break
                with perf.time("plotting"):
                    live.update(stars, iteration=i, sim_time=i * args.dt)
    except KeyboardInterrupt:
        stop_reason = "interrupted"   # second Ctrl+C: stop now, but still save below
    except Exception as exc:          # save what we have, then re-raise at the end
        stop_reason = f"error: {type(exc).__name__}: {exc}"
        error = exc
    finally:
        stop_request.restore()

    # Simulation time = the steps only (live plotting is timed separately)
    simulation_time: float = float(sum(perf.step_times))
    if stop_reason != "finished":
        print(f"Stopped early ({stop_reason}) after {format_thousands(steps_done)} of "
              f"{format_thousands(args.steps)} steps, {time.perf_counter() - program_start:.1f} s "
              f"into a {time_limit:.0f} s limit. Saving what we have.")

    print(f"Final mean distance to centre of mass: {mean_distance_to_centre_of_mass(stars):.3f} pc")
    print(f"Stars: {format_thousands(n_original)} at start, {format_thousands(stars.count)} at end "
          f"({format_thousands(n_original - stars.count)} merged away)")
    print(f"Total momentum at end: {np.round(total_momentum(stars), 10)} M_sun·km/s (starts at 0)")
    print(f"Simulated {format_thousands(steps_done)} steps "
          f"({steps_done * args.dt * MYR_PER_TIME_UNIT:.2f} Myr) in {simulation_time:.2f} s "
          f"({simulation_time / max(steps_done, 1) * 1000:.2f} ms/step)")

    if args.save_final is not None:
        with perf.time("output"):
            pd.DataFrame({
                "x_pc": stars.positions[:, 0], "y_pc": stars.positions[:, 1], "z_pc": stars.positions[:, 2],
                "vx_kms": stars.velocities[:, 0], "vy_kms": stars.velocities[:, 1], "vz_kms": stars.velocities[:, 2],
                "mass_msun": stars.masses,
                "merged_from": stars.members,  # number of original stars in this star
                "sptype": stars.sptypes,       # class of the heaviest original star in it
                "sptype_guessed": stars.sptype_guessed,  # True if that class was drawn at random
            }).to_csv(args.save_final, index=False)
        print(f"Saved final state to {args.save_final}")

    if live is not None:
        print(f"Saved live plot frames in {live.frames_dir}/")

    if args.plot:
        with perf.time("plotting"):
            final_fig: Figure = plot_start_to_end(start_positions, stars)
            final_title: str = (f"{args.data.stem} — {format_thousands(n_original)} stars — "
                                f"Start to end after {format_thousands(steps_done)} iterations · "
                                f"t = {steps_done * args.dt * MYR_PER_TIME_UNIT:.2f} Myr")
            if stars.count < n_original:
                final_title += f" — {format_thousands(stars.count)} after merging"
            if final_fig.canvas.manager is not None:
                final_fig.canvas.manager.set_window_title(final_title)
            saved_plot: Path = save_figure(final_fig, args.plots_dir, final_title)
        print(f"Saved plot: {saved_plot}")

    # Total time stops here: waiting for plot windows to be closed doesn't count
    total_time: float = time.perf_counter() - program_start

    row: Dict[str, Value] = build_results_row(
        args=args, perf=perf, run_started=run_started, total_time=total_time,
        simulation_time=simulation_time, iterations=steps_done,
        n_stars_start=n_original, stars=stars, stop_reason=stop_reason)
    print_performance_summary(row)
    if not args.no_results:
        append_results(args.results, row)
        print(f"Added run '{row['run_name']}' to {args.results}")

    if error is not None:
        raise error

    # Keep the figure(s) on screen until they are closed
    if args.plot or (live is not None and live.is_open()):
        plt.ioff()
        plt.show()


def print_performance_summary(row: Dict[str, Value]) -> None:
    def seconds(key: str) -> str:
        return f"{float(row[key]):8.3f} s"

    status: str = "completed" if row["completed"] else f"stopped early: {row['stop_reason']}"
    print(f"\nPerformance ({row['mode']}, {status}, {row['iterations']} iterations)")
    print(f"  total        {seconds('total_time_s')}")
    print(f"  load         {seconds('load_time_s')}")
    print(f"  simulation   {seconds('simulation_time_s')}")
    print(f"    forces     {seconds('forces_time_s')}")
    print(f"    merge      {seconds('merge_time_s')}")
    print(f"    integrate  {seconds('integrate_time_s')}")
    print(f"  plotting     {seconds('plotting_time_s')}")
    print(f"  output       {seconds('output_time_s')}")
    print(f"  {float(row['interactions_per_s']):.3e} interactions/s · "
          f"~{float(row['est_gflops']):.2f} GFLOP/s · "
          f"model bandwidth {float(row['model_bandwidth_GBps']):.2f} GB/s · "
          f"peak memory {float(row['peak_memory_mb']):.0f} MB")

if __name__ == "__main__":
    main()