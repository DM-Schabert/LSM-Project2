"""Make Gaia subset CSVs on demand, without plotly.

Produces exactly the files GaiaSubsetSelection.py --save would write
(gaia_<method>_<N>.csv, same rows, same order, same columns), so subsets made
here and subsets made there are interchangeable. GaiaSubsetSelection.py
imports plotly at the top, which is not among the cluster modules; this file
only needs numpy and pandas.

If GaiaSubsetSelection.py can be imported (plotly installed), its own
StarSelector is used, so the two can never drift apart. Otherwise the copy
below is used; it mirrors load_gaia_data and StarSelector line by line.
If you change the selection in GaiaSubsetSelection.py, change it here too.

Usage:
    python3 isoeff/make_subset.py --n 1000 2000 4000            # gaia_subsets/gaia_nearest_<N>.csv
    python3 isoeff/make_subset.py --n 5000 --method random
    python3 isoeff/make_subset.py --n 8000 --csv /path/gaia_cleaned.csv --out gaia_subsets
Existing files are left alone (use --force to rewrite them).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_SEED: int = 42  # GaiaSubsetSelection.DEFAULT_SEED


def subset_path(out_dir: Path, method: str, n: int) -> Path:
    return out_dir / f"gaia_{method}_{n}.csv"


# ---------------------------------------------------------------- copy of GaiaSubsetSelection

def _load_gaia_data(csv_path: Path) -> pd.DataFrame:
    """Same as GaiaSubsetSelection.load_gaia_data (without the row count)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Pass --csv <path to gaia_cleaned.csv>.")
    raw: pd.DataFrame = pd.read_csv(csv_path)
    data: pd.DataFrame = raw.loc[raw["Plx"] > 0].copy().reset_index(drop=True)
    data["SpType-ELS"] = data["SpType-ELS"].str.strip()

    ra = np.deg2rad(data["RA_ICRS"].to_numpy())
    dec = np.deg2rad(data["DE_ICRS"].to_numpy())
    distance_pc = 1000.0 / data["Plx"].to_numpy()
    data["x_pc"] = distance_pc * np.cos(dec) * np.cos(ra)
    data["y_pc"] = distance_pc * np.cos(dec) * np.sin(ra)
    data["z_pc"] = distance_pc * np.sin(dec)
    return data


class _StarSelector:
    """Same as GaiaSubsetSelection.StarSelector."""

    def __init__(self, data: pd.DataFrame, seed: int) -> None:
        self.data = data
        rng = np.random.default_rng(seed)
        self.random_order = rng.permutation(len(data))
        xyz = data[["x_pc", "y_pc", "z_pc"]].to_numpy()
        self.center = np.median(xyz, axis=0)
        distance_sq = np.sum((xyz - self.center) ** 2, axis=1)
        self.nearest_order = np.argsort(distance_sq, kind="stable")

    def select(self, n: int, method: str = "random") -> pd.DataFrame:
        if not 1 <= n <= len(self.data):
            raise ValueError(f"N must be between 1 and {len(self.data):,}")
        if method == "random":
            indices = self.random_order[:n]
        elif method == "nearest":
            indices = self.nearest_order[:n]
        else:
            raise ValueError("method must be 'random' or 'nearest'")
        return self.data.iloc[indices].copy()


def _selector(csv_path: Path, seed: int) -> Tuple[object, str]:
    """GaiaSubsetSelection's own selector if it imports, otherwise the copy above."""
    code_dir: Path = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(code_dir))
    try:
        import GaiaSubsetSelection as gss  # needs plotly
        data, _ = gss.load_gaia_data(csv_path)
        return gss.StarSelector(data, seed=seed), "GaiaSubsetSelection.py"
    except ImportError:
        return _StarSelector(_load_gaia_data(csv_path), seed), "isoeff/make_subset.py (copy)"
    finally:
        sys.path.remove(str(code_dir))


# ---------------------------------------------------------------- public

def ensure_subsets(n_values: Sequence[int], csv_path: Path, out_dir: Path, method: str = "nearest",
                   seed: int = DEFAULT_SEED, force: bool = False, quiet: bool = False) -> List[Path]:
    """Write every missing gaia_<method>_<N>.csv. Loads the big CSV only if something is missing."""
    wanted: List[Path] = [subset_path(out_dir, method, n) for n in n_values]
    missing: List[Tuple[int, Path]] = [(n, p) for n, p in zip(n_values, wanted)
                                       if force or not p.exists()]
    if not missing:
        return wanted
    start: float = time.perf_counter()
    selector, source = _selector(csv_path, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, path in missing:
        tmp: Path = path.with_name(path.name + ".tmp")
        selector.select(n, method).to_csv(tmp, index=False)  # type: ignore[attr-defined]
        tmp.replace(path)  # atomic: a half-written file is never picked up as a subset
        if not quiet:
            print(f"Made {path}")
    if not quiet:
        print(f"Subsets made with {source} in {time.perf_counter() - start:.1f} s")
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Gaia subset CSVs (no plotly needed).")
    parser.add_argument("--n", type=int, nargs="+", required=True, help="Sample sizes")
    parser.add_argument("--method", choices=("nearest", "random"), default="nearest")
    parser.add_argument("--csv", type=Path, default=Path("gaia_cleaned.csv"),
                        help="Path to the cleaned Gaia CSV (default: gaia_cleaned.csv)")
    parser.add_argument("--out", type=Path, default=Path("gaia_subsets"),
                        help="Output folder (default: gaia_subsets)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Rewrite files that already exist")
    args = parser.parse_args()
    ensure_subsets(args.n, args.csv, args.out, args.method, args.seed, args.force)


if __name__ == "__main__":
    main()
