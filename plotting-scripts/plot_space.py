
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------
# Settings
# -----------------------------
N_STARS = 600_000
MAX_DISTANCE = None  # Set to 5000 to focus on nearby stars
SEED = 42

# -----------------------------
# 1. Load Gaia data
# -----------------------------
path = Path(__file__).resolve().parent

zip_file = path / "archive (1).zip"
csv_file = path / "dataGaia2.csv"

data_file = zip_file if zip_file.exists() else csv_file

df = pd.read_csv(
    data_file,
    compression="zip" if data_file.suffix == ".zip" else None,
    usecols=["RA_ICRS", "DE_ICRS", "Dist", "Teff"]
)

df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df["Dist"] > 0]

if MAX_DISTANCE is not None:
    df = df[df["Dist"] <= MAX_DISTANCE]

# Random sample for faster plotting
df = df.sample(n=min(N_STARS, len(df)), random_state=SEED)

# -----------------------------
# 2. Convert to 3D coordinates
# -----------------------------
ra = np.radians(df["RA_ICRS"].to_numpy())
dec = np.radians(df["DE_ICRS"].to_numpy())
r = df["Dist"].to_numpy()

x = r * np.cos(dec) * np.cos(ra)
y = r * np.cos(dec) * np.sin(ra)
z = r * np.sin(dec)

# -----------------------------
# 3. Interactive 3D plot
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(
        size=2,
        color=df["Teff"],
        colorscale="Turbo",
        cmin=3500,
        cmax=15000,
        opacity=0.65,
        colorbar=dict(title="Temperature (K)")
    ),
    text=[f"{t:.0f} K" for t in df["Teff"]],
    hovertemplate=(
        "X: %{x:.0f} pc<br>"
        "Y: %{y:.0f} pc<br>"
        "Z: %{z:.0f} pc<br>"
        "Temperature: %{text}<extra></extra>"
    )
))

# Earth/Sun at the origin
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode="markers",
    marker=dict(size=6, color="white"),
    name="Observer (Sun)"
))

fig.update_layout(
    title=f"Gaia 3D Star Distribution ({len(df):,} stars)",
    template="plotly_dark",
    scene=dict(
        xaxis_title="X (pc)",
        yaxis_title="Y (pc)",
        zaxis_title="Z (pc)",
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    showlegend=True
)

# Open interactive plot in browser
fig.write_html("gaia_stars_3d.html", auto_open=True)
