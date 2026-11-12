
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path

# ----------------------------------
# SETTINGS
# ----------------------------------
POINT_SIZE = 1
MAX_DISTANCE = None   # None = include all distances
SMOOTH_RENDERING = True

# ----------------------------------
# LOAD ALL STARS
# ----------------------------------
folder = Path(__file__).resolve().parent
file = folder / "archive (1).zip"

df = pd.read_csv(
    file,
    compression="zip",
    usecols=["RA_ICRS", "DE_ICRS", "Dist", "Teff"],
    dtype={
        "RA_ICRS": "float32",
        "DE_ICRS": "float32",
        "Dist": "float32",
        "Teff": "float32"
    }
)

# Only discard stars without usable positions
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["RA_ICRS", "DE_ICRS", "Dist"])
df = df[df["Dist"] > 0]

if MAX_DISTANCE is not None:
    df = df[df["Dist"] <= MAX_DISTANCE]

print(f"Rendering {len(df):,} stars")

# ----------------------------------
# CARTESIAN COORDINATES
# ----------------------------------
ra = np.deg2rad(df["RA_ICRS"].to_numpy())
dec = np.deg2rad(df["DE_ICRS"].to_numpy())
r = df["Dist"].to_numpy()

x = r * np.cos(dec) * np.cos(ra)
y = r * np.cos(dec) * np.sin(ra)
z = r * np.sin(dec)

points = np.column_stack((x, y, z)).astype("float32")

# Stars with missing temperatures get a neutral temperature
temperature = df["Teff"].to_numpy()
temperature = np.nan_to_num(temperature, nan=6000.0)

# ----------------------------------
# INTERACTIVE 3D VISUALIZATION
# ----------------------------------
plotter = pv.Plotter(
    window_size=(1400, 900),
    title="Gaia Star Distribution"
)

plotter.set_background("#030510")

plotter.add_points(
    points,
    scalars=temperature,
    style="points_gaussian" if SMOOTH_RENDERING else "points",
    point_size=POINT_SIZE,
    render_points_as_spheres=False,
    emissive=True,
    cmap="coolwarm_r",
    clim=[3000, 15000],
    show_scalar_bar=True,
    scalar_bar_args={"title": "Temperature (K)"}
)

# Sun at the origin
plotter.add_mesh(
    pv.Sphere(radius=100, center=(0, 0, 0)),
    color="yellow",
    name="Sun"
)

plotter.add_axes()
plotter.show_grid(
    xtitle="X (pc)",
    ytitle="Y (pc)",
    ztitle="Z (pc)"
)

# Fast anti-aliasing for smoother edges
plotter.enable_anti_aliasing("fxaa")

plotter.add_text(
    f"Gaia | {len(points):,} stars",
    position="upper_left",
    font_size=12,
    color="white"
)

plotter.reset_camera()
plotter.show()