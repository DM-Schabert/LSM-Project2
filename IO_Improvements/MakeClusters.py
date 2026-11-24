"""Clusters of real Gaia stars for the cluster decomposition (Cluster_Plan_v3.md, step 1).

Picks C cluster centres inside the Gaia data and gives each centre its n
nearest stars, so each MPI node later gets one cluster of real stars.

Output: ONE CSV with every original column, x_pc/y_pc/z_pc and a `cluster`
column (0 .. C-1). Rows are sorted by cluster, and within a cluster by distance
to its centre. SerialImplementation.py and the v1/ring MPI versions read it
unchanged (they ignore `cluster`), so the same file gives the direct-summation
reference. A second CSV lists the centres, the cluster radii and the nearest
neighbour of every cluster.

How the centres are chosen (reproducible with --seed):
    1. Centre of the observations: the median 3-D position of all stars, the
       same centre StarSelector uses for the nearest-N subsets. It is also the
       first cluster centre.
    2. Range: the --range-percentile distance of all stars from that centre.
       Candidate centres are drawn uniformly inside a ball of radius range / 2
       around it.
    3. A cluster's radius r(c) is the distance from c to its n-th nearest star.
       A candidate c is accepted when, for every accepted centre a,
           |c - a| >= min_gap * (r(c) + r(a))
       (the clusters don't overlap, and theta = (r_a + r_c) / D <= 1 / min_gap,
       so the far-field approximation holds), and its nearest accepted centre a
       is no further than
           |c - a| <= max_gap * (r(c) + r(a))
       (it sits close enough to feel at least one other cluster).
    4. An attempt that is stuck after --max-tries candidates (the clusters placed
       early can leave no room for the rest) is started again from the first
       centre, up to --restarts times. After that it stops and says what to change.

Theta here uses the outermost star of each cluster, so it is an upper bound on
the theta the simulation computes from rms radii.

All stars start at rest (SerialImplementation.load_subset has no velocities
yet), so each cluster will collapse under its own gravity during a run.

Usage:
    python3 IO_Improvements/MakeClusters.py --clusters 2 --n 1000          # from the project root
    python3 MakeClusters.py --clusters 4 --n 2700 --min-gap 2 --max-gap 5  # from IO_Improvements/
    python3 IO_Improvements/MakeClusters.py --clusters 4 --n 2700 --csv /path/gaia_cleaned.csv

Lives in IO_Improvements/ in the project root. GaiaSubsetSelection.py, isoeff/,
gaia_cleaned.csv and gaia_subsets/ are looked for in the root, so it works
from any folder.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# The project root: this file is in <root>/IO_Improvements/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "gaia_cleaned.csv"
DEFAULT_OUT_DIR: Path = PROJECT_ROOT / "gaia_subsets"
DEFAULT_SEED: int = 42
DEFAULT_RANGE_PERCENTILE: float = 90.0  # the catalogue has some extremely distant stars
DEFAULT_MIN_GAP: float = 2.0            # theta <= 0.5
DEFAULT_MAX_GAP: float = 5.0
DEFAULT_MAX_TRIES: int = 2_000        # candidates per attempt
DEFAULT_RESTARTS: int = 20


# ---------------------------------------------------------------- data

def load_gaia(csv_path: Path) -> pd.DataFrame:
    """The cleaned Gaia data with x_pc, y_pc, z_pc, loaded exactly as for the nearest/random subsets.

    Uses GaiaSubsetSelection.load_gaia_data when plotly is installed, otherwise
    the plotly-free copy in isoeff/make_subset.py (same code).
    """
    root: str = str(PROJECT_ROOT)  # where GaiaSubsetSelection.py and isoeff/ are
    sys.path.insert(0, root)
    try:
        try:
            import GaiaSubsetSelection as gss  # imports plotly at the top
            data, _ = gss.load_gaia_data(csv_path)
            return data
        except ImportError:
            from isoeff.make_subset import _load_gaia_data
            return _load_gaia_data(csv_path)
    finally:
        sys.path.remove(root)


def get_xyz(data: pd.DataFrame) -> FloatArray:
    return np.ascontiguousarray(data[["x_pc", "y_pc", "z_pc"]].to_numpy(dtype=np.float64))


def get_center_of_observations(xyz: FloatArray) -> FloatArray:
    """Median 3-D position: the same centre StarSelector uses for the nearest-N subsets."""
    return np.median(xyz, axis=0)


def get_range(xyz: FloatArray, center: FloatArray, percentile: float) -> float:
    """How far the observations reach: the given percentile of the star distances from the centre."""
    return float(np.percentile(np.linalg.norm(xyz - center, axis=1), percentile))


# ---------------------------------------------------------------- centres

def cluster_radius(xyz: FloatArray, center: FloatArray, n: int) -> float:
    """Distance from `center` to its n-th nearest star: the radius of a cluster of n stars there."""
    distance_sq: FloatArray = np.sum((xyz - center) ** 2, axis=1)
    return float(np.sqrt(np.partition(distance_sq, n - 1)[n - 1]))


def random_point_in_ball(rng: np.random.Generator, center: FloatArray, radius: float) -> FloatArray:
    """Uniform in volume: random direction, radius ∝ u^(1/3)."""
    direction: FloatArray = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return center + direction * radius * rng.random() ** (1 / 3)


def try_cluster_centers(xyz: FloatArray, number_of_clusters: int, n: int, center_of_obs: FloatArray,
                        search_radius: float, min_gap: float, max_gap: float, max_tries: int,
                        rng: np.random.Generator) -> Tuple[List[FloatArray], List[float], int]:
    """One attempt: add random candidates until all clusters are placed or max_tries is used up.

    Returns the centres, their radii and the number of candidates drawn. Fewer
    centres than asked for means this attempt got stuck: the clusters placed
    early can leave no room for the rest.
    """
    centers: List[FloatArray] = [center_of_obs]
    radii: List[float] = [cluster_radius(xyz, center_of_obs, n)]
    tries: int = 0
    while len(centers) < number_of_clusters and tries < max_tries:
        tries += 1
        candidate: FloatArray = random_point_in_ball(rng, center_of_obs, search_radius)
        r_candidate: float = cluster_radius(xyz, candidate, n)
        distances: FloatArray = np.linalg.norm(np.array(centers) - candidate, axis=1)
        reach: FloatArray = r_candidate + np.array(radii)     # r(c) + r(a) for every accepted a
        if np.any(distances < min_gap * reach):                # overlaps, or too close for the far field
            continue
        nearest: int = int(np.argmin(distances))
        if distances[nearest] > max_gap * reach[nearest]:      # too far from every other cluster
            continue
        centers.append(candidate)
        radii.append(r_candidate)
    return centers, radii, tries


def create_cluster_centers(xyz: FloatArray, number_of_clusters: int, n: int, center_of_obs: FloatArray,
                           search_radius: float, min_gap: float, max_gap: float, max_tries: int,
                           restarts: int, rng: np.random.Generator) -> Tuple[List[FloatArray], List[float]]:
    """Centres and radii of `number_of_clusters` clusters of n stars; the first centre is center_of_obs.

    See the module docstring for the acceptance rule. A stuck attempt is
    thrown away and started again from the first centre, up to `restarts` times.
    """
    best: int = 0
    for attempt in range(1, restarts + 1):
        centers, radii, tries = try_cluster_centers(xyz, number_of_clusters, n, center_of_obs,
                                                    search_radius, min_gap, max_gap, max_tries, rng)
        if len(centers) == number_of_clusters:
            print(f"Placed {number_of_clusters} centres in attempt {attempt} ({tries:,} candidates)")
            return centers, radii
        best = max(best, len(centers))
    raise SystemExit(
        f"Placed at most {best} of {number_of_clusters} clusters in {restarts} attempts of "
        f"{max_tries:,} candidates. Try fewer or smaller clusters (--clusters, --n), a smaller "
        f"--min-gap, a larger --max-gap or a larger --range-percentile.")


# ---------------------------------------------------------------- stars

def select_cluster_stars(xyz: FloatArray, centers: List[FloatArray], n: int) -> List[IntArray]:
    """The n nearest stars of every centre, nearest first. The lists never share a star."""
    members: List[IntArray] = []
    for center in centers:
        distance_sq: FloatArray = np.sum((xyz - center) ** 2, axis=1)
        members.append(np.argsort(distance_sq, kind="stable")[:n].astype(np.int64))
    all_members: IntArray = np.concatenate(members)
    if len(np.unique(all_members)) != len(all_members):  # can only happen if min_gap < 1
        raise SystemExit("Two clusters share stars; use --min-gap of at least 1")
    return members


@dataclass
class ClusterInfo:
    cluster: int
    x_pc: float
    y_pc: float
    z_pc: float
    r_half_pc: float       # half the stars are closer to the centre than this
    r_max_pc: float        # outermost star
    nearest_cluster: int
    nearest_distance_pc: float
    theta_nearest: float   # (r_max + r_max of the nearest) / distance; the far field needs this small


def describe_clusters(xyz: FloatArray, centers: List[FloatArray], members: List[IntArray]) -> pd.DataFrame:
    radii: List[Tuple[float, float]] = []
    for center, idx in zip(centers, members):
        distance: FloatArray = np.linalg.norm(xyz[idx] - center, axis=1)
        radii.append((float(np.median(distance)), float(distance.max())))
    c: FloatArray = np.array(centers)
    rows: List[ClusterInfo] = []
    for k, center in enumerate(centers):
        distances: FloatArray = np.linalg.norm(c - center, axis=1)
        distances[k] = np.inf
        nearest: int = int(np.argmin(distances))
        d: float = float(distances[nearest])
        rows.append(ClusterInfo(
            cluster=k, x_pc=float(center[0]), y_pc=float(center[1]), z_pc=float(center[2]),
            r_half_pc=radii[k][0], r_max_pc=radii[k][1],
            nearest_cluster=nearest if np.isfinite(d) else -1,
            nearest_distance_pc=d if np.isfinite(d) else float("nan"),
            theta_nearest=(radii[k][1] + radii[nearest][1]) / d if np.isfinite(d) else 0.0))
    return pd.DataFrame([vars(row) for row in rows])


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp: Path = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)  # a half-written file is never picked up


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clusters of real Gaia stars, one cluster per MPI node.")
    parser.add_argument("--clusters", type=int, required=True, help="Number of clusters C (= nodes)")
    parser.add_argument("--n", type=int, required=True, help="Stars per cluster")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH,
                        help="Cleaned Gaia CSV (default: gaia_cleaned.csv in the project root)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help="Output folder (default: gaia_subsets/ in the project root)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--range-percentile", type=float, default=DEFAULT_RANGE_PERCENTILE,
                        help="Centres are drawn within half this distance percentile of the centre "
                             f"of the observations (default: {DEFAULT_RANGE_PERCENTILE:g})")
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP,
                        help="Centre distance >= min_gap · (r_a + r_b); theta <= 1/min_gap "
                             f"(default: {DEFAULT_MIN_GAP:g})")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP,
                        help="Every cluster has a neighbour within max_gap · (r_a + r_b) "
                             f"(default: {DEFAULT_MAX_GAP:g})")
    parser.add_argument("--max-tries", type=int, default=DEFAULT_MAX_TRIES,
                        help=f"Candidates per attempt (default: {DEFAULT_MAX_TRIES:,})")
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS,
                        help=f"Attempts before giving up (default: {DEFAULT_RESTARTS})")
    parser.add_argument("--force", action="store_true", help="Rewrite the files if they exist")
    args = parser.parse_args()
    if args.clusters < 1 or args.n < 1:
        parser.error("--clusters and --n must be at least 1")
    if not 1.0 <= args.min_gap <= args.max_gap:
        parser.error("need 1 <= --min-gap <= --max-gap")
    return args


def main() -> None:
    args = parse_args()
    out_path: Path = args.out / f"gaia_clusters_C{args.clusters}_n{args.n}.csv"
    centres_path: Path = args.out / f"gaia_clusters_C{args.clusters}_n{args.n}_centres.csv"
    if out_path.exists() and not args.force:
        print(f"{out_path} exists (use --force to rewrite it)")
        return

    start: float = time.perf_counter()
    data: pd.DataFrame = load_gaia(args.csv)
    xyz: FloatArray = get_xyz(data)
    if args.clusters * args.n > len(data):
        raise SystemExit(f"{args.clusters} × {args.n:,} stars is more than the {len(data):,} in the data")

    center_of_obs: FloatArray = get_center_of_observations(xyz)
    reach: float = get_range(xyz, center_of_obs, args.range_percentile)
    print(f"{len(data):,} stars; centre of the observations {np.round(center_of_obs, 2)} pc; "
          f"{args.range_percentile:g}% of the stars within {reach:.1f} pc of it")

    rng: np.random.Generator = np.random.default_rng(args.seed)
    centers, _ = create_cluster_centers(xyz, args.clusters, args.n, center_of_obs, reach / 2,
                                        args.min_gap, args.max_gap, args.max_tries, args.restarts, rng)
    members: List[IntArray] = select_cluster_stars(xyz, centers, args.n)

    clusters: pd.DataFrame = pd.concat(
        [data.iloc[idx].assign(cluster=k) for k, idx in enumerate(members)], ignore_index=True)
    info: pd.DataFrame = describe_clusters(xyz, centers, members)

    args.out.mkdir(parents=True, exist_ok=True)
    write_atomic(clusters, out_path)
    write_atomic(info, centres_path)

    print(info.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"Largest theta between neighbours: {info['theta_nearest'].max():.2f} "
          f"(the far-field approximation wants it well below 1)")
    print(f"Saved {out_path} ({len(clusters):,} stars) and {centres_path} "
          f"in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()