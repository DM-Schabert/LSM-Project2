import pandas as pd, numpy as np, pyvista as pv
from astropy.coordinates import SkyCoord
import astropy.units as u

df = pd.read_csv("dataGaia2.csv")
cols = {c.lower().replace("-", "_"): c for c in df.columns}   # case-insensitive lookup
RA, DEC, PLX = cols["ra_icrs"], cols["de_icrs"], cols["plx"]
COLOUR = cols.get("bp_rp")

df = df[df[PLX] > 1].dropna(subset=[RA, DEC, PLX])
c = SkyCoord(ra=df[RA].values*u.deg, dec=df[DEC].values*u.deg,
             distance=(1000/df[PLX].values)*u.pc).galactic
xyz = np.column_stack([c.cartesian.x.value, c.cartesian.y.value, c.cartesian.z.value])

cloud = pv.PolyData(xyz)
lc = {c.lower(): c for c in df.columns}
print(df.columns.tolist())

if "bp-rp" in lc or "bp_rp" in lc:
    colour, cmap, clim = df[lc.get("bp-rp", lc.get("bp_rp"))], "RdYlBu_r", [-0.5, 3]
elif "bpmag" in lc and "rpmag" in lc:
    colour, cmap, clim = df[lc["bpmag"]] - df[lc["rpmag"]], "RdYlBu_r", [-0.5, 3]
elif "teff" in lc:
    colour, cmap, clim = np.log10(df[lc["teff"]]), "RdYlBu", [np.log10(3000), np.log10(20000)]
else:
    sp = next(c for c in df.columns if c.lower().startswith(("sptype", "class", "spec")))
    colour = df[sp].astype(str).str[0].map({"O":0,"B":1,"A":2,"F":3,"G":4,"K":5,"M":6})
    cmap, clim = "RdYlBu_r", [0, 6]
print("colouring by:", colour.name)
cloud["colour"] = colour.values
p = pv.Plotter()
p.set_background("black")
p.add_points(cloud, scalars="colour", cmap="RdYlBu_r", point_size=1, opacity=0.6,
             clim=[-0.5, 3])   # keeps a few extreme colours from washing out the rest
p.show()