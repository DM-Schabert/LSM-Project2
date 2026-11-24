"""Serial reference for the cluster far field (Cluster_Plan_v3.md, step 3). No MPI.

Two commands:

    forces    How good is the approximation? At t = 0, compare the accelerations
              of the approximation (own cluster directly + every other cluster
              as one point mass) with direct summation over all stars.
              --scale moves the clusters apart (> 1) or together (< 1) around
              their common centre of mass; every star keeps its place within
              its cluster. One file then gives the error against separation.

                python3 IO_Improvements/cluster_reference.py forces gaia_subsets/gaia_clusters_C2_n1000.csv \\
                    --scale 0.25 0.5 1 2 4 --out force_error_C2_n1000.csv

              Per scale it prints the largest theta, how large the far field
              is compared with the total force, and the relative error of the
              approximation (median, 99th percentile, max), next to the error
              of leaving the far field out altogether.

    simulate  The approximation, stepped in one process, in the same order as
              ClusterImplementation.py --far-field monopole: cluster summaries
              from the positions at the start of the step, each cluster's own
              forces and merges, then the far field, kick and drift. Its final
              state is the reference for the MPI code:

                python3 IO_Improvements/cluster_reference.py simulate gaia_subsets/gaia_clusters_C2_n1000.csv \\
                    --steps 300 --merge-radius 0.001 --save-final ref_far.csv
                python3 IO_Improvements/compare_clusters.py compare final_far.csv ref_far.csv --tol 1e-12

              The MPI code adds each rank's partial sums for a cluster's
              summary in another order than one np.sum here, so the two agree
              to round-off (about 1e-16), not bit for bit.

The monopole formula is ClusterImplementation.monopole_accelerations, so both
use exactly the same far-field kernel; the force check against direct
summation is what tests the formula itself.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
for folder in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
import pandas as pd

import SerialImplementation as serial
from ClusterImplementation import ClusterSummaries, monopole_accelerations, summary_row
from SerialImplementation import FloatArray, IntArray, Stars


def load(path: Path, seed: int) -> Tuple[Stars, IntArray]:
    """The stars as SerialImplementation loads them, and every star's cluster label."""
    stars: Stars = serial.load_subset(path, np.random.default_rng(seed))
    cluster_of: IntArray = pd.read_csv(path, usecols=["cluster"])["cluster"].to_numpy(dtype=np.int64)
    return stars, cluster_of


def summaries_of(clusters: List[Stars]) -> ClusterSummaries:
    return ClusterSummaries.from_sums(np.array([summary_row(c.positions, c.masses) for c in clusters]))


def split_clusters(stars: Stars, cluster_of: IntArray) -> List[Stars]:
    """One Stars object per cluster, with local indices (as ClusterImplementation.stars_of_cluster)."""
    out: List[Stars] = []
    for k in range(int(cluster_of.max()) + 1):
        rows: IntArray = np.nonzero(cluster_of == k)[0]
        out.append(Stars(positions=np.ascontiguousarray(stars.positions[rows]),
                         velocities=np.ascontiguousarray(stars.velocities[rows]),
                         masses=stars.masses[rows].copy(), members=stars.members[rows].copy(),
                         owner=np.arange(len(rows), dtype=np.int64),
                         sptypes=stars.sptypes[rows].copy(), sptype_guessed=stars.sptype_guessed[rows].copy()))
    return out


# ---------------------------------------------------------------- forces

def force_error(stars: Stars, cluster_of: IntArray, softening: float, scale: float) -> dict:
    """Accelerations at t = 0 with the clusters `scale` times as far apart: approximation vs direct summation."""
    clusters_at_1: List[Stars] = split_clusters(stars, cluster_of)
    centres: FloatArray = summaries_of(clusters_at_1).centres
    common: FloatArray = np.average(stars.positions, axis=0, weights=stars.masses)
    positions: FloatArray = stars.positions + (scale - 1.0) * (centres[cluster_of] - common)

    direct, _ = serial.compute_accelerations(positions, stars.masses, softening, 0.0)

    moved: Stars = Stars(positions=positions, velocities=stars.velocities, masses=stars.masses,
                         members=stars.members, owner=stars.owner, sptypes=stars.sptypes,
                         sptype_guessed=stars.sptype_guessed)
    clusters: List[Stars] = split_clusters(moved, cluster_of)
    summaries: ClusterSummaries = summaries_of(clusters)
    own: FloatArray = np.empty_like(positions)
    far: FloatArray = np.empty_like(positions)
    for k, cluster in enumerate(clusters):
        rows: IntArray = np.nonzero(cluster_of == k)[0]
        own[rows], _ = serial.compute_accelerations(cluster.positions, cluster.masses, softening, 0.0)
        far[rows] = monopole_accelerations(cluster.positions, summaries.centres, summaries.masses, k, softening)

    size: FloatArray = np.linalg.norm(direct, axis=1)
    error: FloatArray = np.linalg.norm(own + far - direct, axis=1) / size
    error_without: FloatArray = np.linalg.norm(own - direct, axis=1) / size
    far_share: FloatArray = np.linalg.norm(far, axis=1) / np.linalg.norm(own + far, axis=1)
    distance: FloatArray = np.linalg.norm(summaries.centres[:, None] - summaries.centres[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    return {
        "scale": scale,
        "min_distance_pc": float(distance.min()),
        "theta_max": float(summaries.theta().max()),
        "far_share_median": float(np.median(far_share)),
        "error_median": float(np.median(error)),
        "error_p99": float(np.percentile(error, 99)),
        "error_max": float(error.max()),
        "error_without_far_median": float(np.median(error_without)),
        "error_without_far_max": float(error_without.max()),
    }


def run_forces(args: argparse.Namespace) -> None:
    stars, cluster_of = load(args.data, args.seed)
    print(f"{stars.count:,} stars in {int(cluster_of.max()) + 1} clusters from {args.data}; "
          f"direct summation over all of them for every scale")
    rows: List[dict] = []
    for scale in args.scale:
        start: float = time.perf_counter()
        rows.append(force_error(stars, cluster_of, args.softening, scale))
        print(f"  scale {scale:g} done in {time.perf_counter() - start:.1f} s")
    table: pd.DataFrame = pd.DataFrame(rows)
    shown: pd.DataFrame = table.rename(columns={
        "min_distance_pc": "closest pair of centres (pc)", "theta_max": "largest theta",
        "far_share_median": "far field / total (median)", "error_median": "error median",
        "error_p99": "error 99%", "error_max": "error max",
        "error_without_far_median": "error without far field (median)",
        "error_without_far_max": "error without far field (max)"})
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("error = |a_approx − a_direct| / |a_direct| per star; 'without far field' leaves the other clusters out")
    if args.out is not None:
        table.to_csv(args.out, index=False)
        print(f"Saved {args.out}")


# ---------------------------------------------------------------- simulate

def step_all(clusters: List[Stars], args: argparse.Namespace) -> List[Stars]:
    """One step of every cluster, in the same order as ClusterImplementation.cluster_step."""
    summaries: ClusterSummaries = summaries_of(clusters)  # from the positions before the step
    stepped: List[Stars] = []
    for k, cluster in enumerate(clusters):
        accelerations, pairs = serial.compute_accelerations(cluster.positions, cluster.masses,
                                                            args.softening, args.merge_radius)
        while pairs:  # the cluster's own merges, as serial.step
            cluster = serial.merge_stars(cluster, pairs)
            accelerations, pairs = serial.compute_accelerations(cluster.positions, cluster.masses,
                                                                args.softening, args.merge_radius)
        if len(clusters) > 1:
            accelerations += monopole_accelerations(cluster.positions, summaries.centres, summaries.masses,
                                                    k, args.softening)
        cluster.velocities += accelerations * args.dt
        cluster.positions += cluster.velocities * args.dt
        stepped.append(cluster)
    return stepped


def run_simulate(args: argparse.Namespace) -> None:
    stars, cluster_of = load(args.data, args.seed)
    clusters: List[Stars] = split_clusters(stars, cluster_of)
    print(f"{stars.count:,} stars in {len(clusters)} clusters; {args.steps} steps with the monopole far field")
    start: float = time.perf_counter()
    for i in range(1, args.steps + 1):
        before: int = sum(c.count for c in clusters)
        clusters = step_all(clusters, args)
        after: int = sum(c.count for c in clusters)
        if after < before:
            print(f"Iteration {i}: {before - after} star(s) merged, {after} left")
    print(f"Done in {time.perf_counter() - start:.1f} s; stars per cluster at the end: "
          + ", ".join(str(c.count) for c in clusters))
    if args.save_final is not None:
        pd.DataFrame({
            "x_pc": np.concatenate([c.positions[:, 0] for c in clusters]),
            "y_pc": np.concatenate([c.positions[:, 1] for c in clusters]),
            "z_pc": np.concatenate([c.positions[:, 2] for c in clusters]),
            "vx_kms": np.concatenate([c.velocities[:, 0] for c in clusters]),
            "vy_kms": np.concatenate([c.velocities[:, 1] for c in clusters]),
            "vz_kms": np.concatenate([c.velocities[:, 2] for c in clusters]),
            "mass_msun": np.concatenate([c.masses for c in clusters]),
            "merged_from": np.concatenate([c.members for c in clusters]),
            "sptype": np.concatenate([c.sptypes for c in clusters]),
            "sptype_guessed": np.concatenate([c.sptype_guessed for c in clusters]),
            "cluster": np.repeat(np.arange(len(clusters)), [c.count for c in clusters]),
        }).to_csv(args.save_final, index=False)
        print(f"Saved final state to {args.save_final}")


# ---------------------------------------------------------------- command line

def main() -> None:
    parser = argparse.ArgumentParser(description="Serial reference for the cluster far field (no MPI).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("forces", "Force error at t = 0 against direct summation"),
                       ("simulate", "Step the approximation in one process")):
        p = sub.add_parser(name, help=text)
        p.add_argument("data", type=Path, help="CSV with a 'cluster' column")
        p.add_argument("--softening", type=serial.non_negative_float, default=0.01,
                       help="Softening length in pc (default: 0.01)")
        p.add_argument("--seed", type=int, default=serial.DEFAULT_SEED)
        if name == "forces":
            p.add_argument("--scale", type=float, nargs="+", default=[1.0],
                           help="Distance factors between the clusters (default: 1 = as in the file)")
            p.add_argument("--out", type=Path, default=None, help="Also save the table as CSV")
        else:
            p.add_argument("--steps", type=serial.positive_int, default=100)
            p.add_argument("--dt", type=float, default=0.01)
            p.add_argument("--merge-radius", type=serial.non_negative_float, default=0.01)
            p.add_argument("--save-final", type=Path, default=None)
    args = parser.parse_args()
    run_forces(args) if args.command == "forces" else run_simulate(args)


if __name__ == "__main__":
    main()
