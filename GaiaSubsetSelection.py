"""Gaia star subsets for the MPI N-body project.

Compares random sampling and nearest-N sampling of the cleaned Gaia data.
The aim is to create reproducible inputs for MPI benchmarks, not physically
realistic stellar systems.

Steps:
    1. Load data and calculate 3D coordinates (parsecs).
    2. Select stars using two methods (random / nearest to the median position).
       For both methods, smaller samples are nested inside larger ones.
    3. Print a distance summary for each selection.
    4. Show interactive 3D Plotly comparisons (opens in the browser).
    5. Optionally export the subsets as CSV files.

MPI benchmark reminder:
    For strong scaling, keep the same selected stars and N and vary the MPI
    rank count. For direct summation, work per time step is O(N^2) regardless
    of sampling method. Numerical validation should compare the MPI result with
    the serial result for identical initial conditions.

Usage examples:
    python GaiaSubsetSelection.py
    python GaiaSubsetSelection.py --n 1000 5000 --no-plot --save
    python GaiaSubsetSelection.py --no-show            # save the plots without opening the browser
    python GaiaSubsetSelection.py --csv data/gaia_cleaned.csv --out my_subsets

Plots are saved in the Plots folder (change with --plots-dir) as interactive
HTML, plus PNG when kaleido is installed. They have no titles; the file name is
what the title would have been.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

Method = Literal["random", "nearest"]
METHODS: Tuple[Method, Method] = ("random", "nearest")

DEFAULT_CSV_PATH: Path = Path("gaia_cleaned.csv")
DEFAULT_SEED: int = 42
DEFAULT_N_VALUES: List[int] = [1_000, 5_000, 20_000]
DEFAULT_OUT_DIR: Path = Path("gaia_subsets")
DEFAULT_PLOTS_DIR: Path = Path("Plots")


# ---------------------------------------------------------------- 1. Load data

def load_gaia_data(csv_path: Path) -> Tuple[pd.DataFrame, int]:
    """Load the cleaned Gaia CSV and add heliocentric Cartesian coordinates.

    RA_ICRS and DE_ICRS are angles in degrees. With parallax Plx in
    milliarcseconds, distance is d = 1000 / Plx parsecs. Only rows with
    non-positive parallax are removed, since they cannot be converted this way.

    Returns the cleaned data and the number of rows in the original file.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Put gaia_cleaned.csv next to this script "
                                f"or pass --csv <path>.")

    raw: pd.DataFrame = pd.read_csv(csv_path)
    data: pd.DataFrame = raw.loc[raw["Plx"] > 0].copy().reset_index(drop=True)
    data["SpType-ELS"] = data["SpType-ELS"].str.strip()

    ra: NDArray[np.float64] = np.deg2rad(data["RA_ICRS"].to_numpy())
    dec: NDArray[np.float64] = np.deg2rad(data["DE_ICRS"].to_numpy())
    distance_pc: NDArray[np.float64] = 1000.0 / data["Plx"].to_numpy()

    # Cartesian coordinates (parsecs)
    data["x_pc"] = distance_pc * np.cos(dec) * np.cos(ra)
    data["y_pc"] = distance_pc * np.cos(dec) * np.sin(ra)
    data["z_pc"] = distance_pc * np.sin(dec)

    return data, len(raw)


def get_xyz(df: pd.DataFrame) -> NDArray[np.float64]:
    """Return the (n, 3) array of Cartesian positions in parsecs."""
    return df[["x_pc", "y_pc", "z_pc"]].to_numpy()


# ---------------------------------------------------------------- 2. Select stars

class StarSelector:
    """Reproducible random and nearest-N selections of the Gaia data.

    The orderings are built once and reused for every N, so smaller samples
    are always nested inside larger ones.
    """

    def __init__(self, data: pd.DataFrame, seed: int) -> None:
        self.data: pd.DataFrame = data

        # Random: a fixed seed gives the same stars every time.
        rng: np.random.Generator = np.random.default_rng(seed)
        self.random_order: NDArray[np.int64] = rng.permutation(len(data))

        # Nearest: the N stars closest to one fixed centre (the median 3D position).
        xyz: NDArray[np.float64] = get_xyz(data)
        self.center: NDArray[np.float64] = np.median(xyz, axis=0)
        distance_sq: NDArray[np.float64] = np.sum((xyz - self.center) ** 2, axis=1)
        self.nearest_order: NDArray[np.int64] = np.argsort(distance_sq, kind="stable")

    def select(self, n: int, method: Method = "random") -> pd.DataFrame:
        """Return n stars from the cleaned data, using a reproducible ordering."""
        if not 1 <= n <= len(self.data):
            raise ValueError(f"N must be between 1 and {len(self.data):,}")
        if method == "random":
            indices: NDArray[np.int64] = self.random_order[:n]
        elif method == "nearest":
            indices = self.nearest_order[:n]
        else:
            raise ValueError("method must be 'random' or 'nearest'")
        return self.data.iloc[indices].copy()


# ---------------------------------------------------------------- 3. Check selections

def summarise_selections(selector: StarSelector, n_values: List[int]) -> pd.DataFrame:
    """Distance-to-centre summary showing how widely spread each selection is."""
    rows: List[Dict[str, Union[int, str]]] = []
    for n in n_values:
        for method in METHODS:
            subset: pd.DataFrame = selector.select(n, method)
            radius_pc: NDArray[np.float64] = np.linalg.norm(get_xyz(subset) - selector.center, axis=1)
            rows.append({
                "N": n,
                "method": method,
                "median distance to centre (pc)": round(float(np.median(radius_pc))),
                "95th percentile distance (pc)": round(float(np.percentile(radius_pc, 95))),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. Visualise

def compare_subsets(selector: StarSelector, n: int, random_plot_percentile: float = 99) -> go.Figure:
    """Build interactive 3D random vs. nearest-N scatter plots.

    Because the dataset contains some extremely distant stars, the random plot
    only hides its most distant (100 - random_plot_percentile)% by default.
    This affects the plot, not the selected dataset. Use 100 to show every star.
    """
    fig: go.Figure = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.04,
    )

    panels: List[Tuple[int, Method, str]] = [
        (1, "random", "#51B8E8"),
        (2, "nearest", "#F19C52"),
    ]
    for column, method, color in panels:
        xyz: NDArray[np.float64] = get_xyz(selector.select(n, method))

        if method == "random" and random_plot_percentile < 100:
            radial_distance: NDArray[np.float64] = np.linalg.norm(xyz - selector.center, axis=1)
            limit: float = float(np.percentile(radial_distance, random_plot_percentile))
            xyz = xyz[radial_distance <= limit]

        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker=dict(size=1.8, color=color, opacity=0.7),
                # No subplot titles, so the legend tells the two panels apart
                name=f"{method.capitalize()} sample (left)" if column == 1
                else f"{method.capitalize()} sample (right)",
                hovertemplate="x: %{x:.0f} pc<br>y: %{y:.0f} pc<br>z: %{z:.0f} pc<extra></extra>",
                showlegend=True,
            ),
            row=1, col=column,
        )

    # Each panel auto-zooms independently, so compare spatial scales as well as shapes.
    fig.update_scenes(
        xaxis_title="x (pc)", yaxis_title="y (pc)", zaxis_title="z (pc)",
        aspectmode="data",
    )
    fig.update_layout(
        template="plotly_dark", height=540, width=1100,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.0, itemsizing="constant"),
    )
    return fig


def compare_subsets_title(n: int) -> str:
    """What the plot's title would be; used as its file name instead."""
    return f"Gaia selection methods — {n:,} stars each"


def title_to_filename(title: str) -> str:
    """'Gaia selection methods — 1,000 stars each' -> 'Gaia_selection_methods_1_000_stars_each'."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", title).strip("_")


def save_plot(fig: go.Figure, folder: Path, title: str) -> List[Path]:
    """Save the figure as interactive HTML and, if kaleido is installed, as PNG."""
    folder.mkdir(parents=True, exist_ok=True)
    stem: str = title_to_filename(title)
    saved: List[Path] = []

    html_path: Path = folder / f"{stem}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    saved.append(html_path)

    png_path: Path = folder / f"{stem}.png"
    try:
        fig.write_image(png_path, scale=2)
        saved.append(png_path)
    except Exception as error:  # kaleido missing or no Chrome for kaleido to use
        print(f"Could not save {png_path.name} ({type(error).__name__}). "
              f"For PNGs: pip install kaleido && plotly_get_chrome")
    return saved


# ---------------------------------------------------------------- 5. Export

def save_subsets(selector: StarSelector, n_values: List[int], out_dir: Path) -> List[Path]:
    """Save CSVs for the serial and MPI implementations.

    They include all original columns plus x_pc, y_pc and z_pc. The same files
    should be reused across parallel benchmarks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for n in n_values:
        for method in METHODS:
            filename: Path = out_dir / f"gaia_{method}_{n}.csv"
            selector.select(n, method).to_csv(filename, index=False)
            print("Saved:", filename)
            saved.append(filename)
    return saved


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create reproducible random / nearest-N Gaia subsets for the MPI N-body project.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH,
                        help=f"Path to the cleaned Gaia CSV (default: {DEFAULT_CSV_PATH})")
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_N_VALUES,
                        help=f"Sample sizes (default: {' '.join(map(str, DEFAULT_N_VALUES))})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the interactive 3D plots (nothing is shown or saved)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plots but don't open them in the browser")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR,
                        help=f"Folder the plots are saved in (default: {DEFAULT_PLOTS_DIR})")
    parser.add_argument("--random-plot-percentile", type=float, default=99,
                        help="Hide random-sample stars beyond this distance percentile in the plot "
                             "(100 shows all; default: 99)")
    parser.add_argument("--save", action="store_true",
                        help="Export the subsets as CSV files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output folder for --save (default: {DEFAULT_OUT_DIR})")
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    # 1. Load data and calculate 3D coordinates
    data, n_raw = load_gaia_data(args.csv)
    print(f"Original rows: {n_raw:,}")
    print(f"Valid 3D positions: {len(data):,} (excluded {n_raw - len(data):,} non-positive parallaxes)")
    print(data.head())
    print()

    # 2. Select stars using two methods
    selector: StarSelector = StarSelector(data, seed=args.seed)
    print("Fixed selection centre (pc):", np.round(selector.center, 2))
    print()

    # 3. Check the selections
    print(summarise_selections(selector, args.n).to_string(index=False))
    print()

    # 4. Interactive 3D visualisation (one browser tab per sample size)
    if not args.no_plot:
        for n in args.n:
            fig: go.Figure = compare_subsets(selector, n, args.random_plot_percentile)
            for path in save_plot(fig, args.plots_dir, compare_subsets_title(n)):
                print("Saved plot:", path)
            if not args.no_show:
                fig.show()

    # 5. Export subsets
    if args.save:
        save_subsets(selector, args.n, args.out)


if __name__ == "__main__":
    main()