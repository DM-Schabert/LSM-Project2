"""Step 2 check (Cluster_Plan_v3.md): with the far field off, every cluster of a
ClusterImplementation.py run must equal a run on that cluster's stars alone.

Two commands:

    split   one CSV per cluster, next to the input (the cluster's rows, in file order)
            python3 IO_Improvements/compare_clusters.py split gaia_subsets/gaia_clusters_C2_n1000.csv
            -> gaia_subsets/gaia_clusters_C2_n1000_k0.csv, gaia_subsets/gaia_clusters_C2_n1000_k1.csv

    compare the clusters run against one reference run per cluster, in cluster order
            python3 IO_Improvements/compare_clusters.py compare final_clusters.csv final_k0.csv final_k1.csv
            final_clusters.csv: --save-final of ClusterImplementation.py (has a 'cluster' column)
            final_k<k>.csv:     --save-final of SerialImplementation.py (or v1) on cluster k's file
            (or ONE file with a cluster column, e.g. cluster_reference.py simulate --save-final)

compare checks positions, velocities, masses and merged_from. With --tile-rows 1
the clusters run must be bit-identical (the default --tol 0); with tiling, pass
a small tolerance such as --tol 1e-12. The spectral classes are not compared:
stars without one get a random class, and those draws depend on which stars are
in the file. Exit code 0 means PASSED, 1 means FAILED.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

COMPARED: List[str] = ["x_pc", "y_pc", "z_pc", "vx_kms", "vy_kms", "vz_kms", "mass_msun", "merged_from"]


def split(path: Path) -> List[Path]:
    """Copy each cluster's lines of the CSV, as text, into a file of its own.

    The lines are copied character for character instead of being read and
    written again with pandas: pandas' default CSV reader can be one bit off in
    the last digit, so a rewritten file would load as slightly different
    numbers, and the comparison could never be bit-identical.
    """
    cluster: np.ndarray = pd.read_csv(path, usecols=["cluster"])["cluster"].to_numpy()
    with open(path, newline="") as handle:
        header, *lines = handle.readlines()
    if len(lines) != len(cluster):  # a quoted field with a line break would break the line-per-star copy
        sys.exit(f"{path}: {len(lines)} lines but {len(cluster)} stars; can't copy lines one per star")
    written: List[Path] = []
    for k in sorted(np.unique(cluster)):
        out: Path = path.with_name(f"{path.stem}_k{int(k)}.csv")
        with open(out, "w", newline="") as handle:
            handle.write(header)
            handle.writelines(line for line, c in zip(lines, cluster) if c == k)
        print(f"Saved {out} ({int((cluster == k).sum()):,} stars)")
        written.append(out)
    return written


def relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(float(np.abs(a).max()), 1e-300)) if a.size else 0.0


def compare(clusters_path: Path, reference_paths: List[Path], tol: float) -> bool:
    """Compare per cluster. The reference is one file per cluster, or one file with a 'cluster'
    column for all of them (e.g. cluster_reference.py simulate --save-final)."""
    run: pd.DataFrame = pd.read_csv(clusters_path)
    n_clusters: int = int(run["cluster"].max()) + 1
    references: List[pd.DataFrame]
    if len(reference_paths) == 1 and "cluster" in pd.read_csv(reference_paths[0], nrows=0).columns:
        combined: pd.DataFrame = pd.read_csv(reference_paths[0])
        references = [combined[combined["cluster"] == k].reset_index(drop=True) for k in range(n_clusters)]
        reference_paths = reference_paths * n_clusters
    elif len(reference_paths) == n_clusters:
        references = [pd.read_csv(path) for path in reference_paths]
    else:
        sys.exit(f"{clusters_path} has {n_clusters} clusters, but {len(reference_paths)} reference files were given")
    all_ok: bool = True
    for k, (ref_path, ref) in enumerate(zip(reference_paths, references)):
        mine: pd.DataFrame = run[run["cluster"] == k].reset_index(drop=True)
        if len(mine) != len(ref):
            print(f"cluster {k}: FAILED, {len(mine)} stars here vs {len(ref)} in {ref_path}")
            all_ok = False
            continue
        a: np.ndarray = mine[COMPARED].to_numpy(dtype=np.float64)
        b: np.ndarray = ref[COMPARED].to_numpy(dtype=np.float64)
        identical: bool = bool(np.array_equal(a, b))
        pos: float = relative_difference(b[:, 0:3], a[:, 0:3])
        vel: float = relative_difference(b[:, 3:6], a[:, 3:6])
        same_merges: bool = bool(np.array_equal(a[:, 7], b[:, 7]))
        ok: bool = identical or (same_merges and pos <= tol and vel <= tol)
        all_ok &= ok
        print(f"cluster {k}: {len(mine):,} stars, identical={identical}, max relative difference: "
              f"positions {pos:.2e}, velocities {vel:.2e}, same merges={same_merges} "
              f"-> {'PASSED' if ok else 'FAILED'}  ({ref_path})")
    print(f"Clusters validation (tolerance {tol:g}): {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a clusters CSV, or compare a clusters run per cluster.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_split = sub.add_parser("split", help="Write one CSV per cluster next to the input")
    p_split.add_argument("data", type=Path)
    p_cmp = sub.add_parser("compare", help="Compare a clusters run with one reference run per cluster")
    p_cmp.add_argument("clusters_final", type=Path, help="--save-final of ClusterImplementation.py")
    p_cmp.add_argument("references", type=Path, nargs="+", help="--save-final of each cluster alone, in order")
    p_cmp.add_argument("--tol", type=float, default=0.0,
                       help="Largest allowed relative difference (default 0: bit-identical)")
    args = parser.parse_args()
    if args.command == "split":
        split(args.data)
    else:
        sys.exit(0 if compare(args.clusters_final, args.references, args.tol) else 1)


if __name__ == "__main__":
    main()
