"""nisar_baselines.py -- interferometric baselines of the two NISAR pairs, from
the GUNW metadata cubes (referee 2: report the spatial baseline).

For each pair the GUNW radarGrid cube gives the perpendicular and parallel
baseline, incidence angle and reference slant range on a coarse map grid at a
set of ellipsoid heights. Over the ASO lidar AOI, at the cube height nearest
the basin's mid elevation, we report:

  B_perp, B_par          median and range (m)
  height of ambiguity    h_a = lambda R sin(theta) / (2 |B_perp|)          (m)
  critical baseline      B_c = lambda R B_w tan(theta) / c  (flat terrain) (m)
  geometric decorrelation  gamma_geom = 1 - |B_perp| / B_c   (Zebker & Villasenor 1992)

gamma_geom is also given for a slope facing the radar (local incidence 10 deg),
the worst realistic case outside layover. Wavelength and range bandwidth are
read from the product.

Run with the nisar_pytools env and TC_DATA_ROOT pointing at the data tree:
    python code/stats/nisar_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nisar"))
from calibrate_enl import GUNWS  # noqa: E402   the two paper GUNWs

# ASO Tuolumne AOI (EPSG:4326), same box the download scripts use
AOI = [-120.288, 37.714, -119.191, 38.241]      # W, S, E, N
CUBE_HEIGHT = 2500.0                             # m, near the basin's mid elevation
C = 299792458.0
PAIRS = {"P12": "20251207", "P23": "20251219"}   # label -> reference date token


def aoi_mask(x, y, epsg):
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform([AOI[0], AOI[2], AOI[0], AOI[2]], [AOI[1], AOI[1], AOI[3], AOI[3]])
    X, Y = np.meshgrid(x, y)
    return (X >= min(xs)) & (X <= max(xs)) & (Y >= min(ys)) & (Y <= max(ys))


def baselines(gunw: Path) -> dict:
    with h5py.File(gunw) as h:
        g = h["science/LSAR/GUNW/metadata/radarGrid"]
        x, y, z = g["xCoordinates"][()], g["yCoordinates"][()], g["heightAboveEllipsoid"][()]
        k = int(np.argmin(np.abs(z - CUBE_HEIGHT)))
        m = aoi_mask(x, y, int(g["projection"][()]))
        bperp = g["perpendicularBaseline"][k][m]
        bpar = g["parallelBaseline"][k][m]
        inc = np.radians(g["incidenceAngle"][k][m])
        rng = g["referenceSlantRange"][k][m]
        lam = C / float(h["science/LSAR/GUNW/grids/frequencyA/centerFrequency"][()])
        bw = float(h["science/LSAR/GUNW/metadata/processingInformation/parameters/reference/frequencyA/rangeBandwidth"][()])
    h_amb = lam * rng * np.sin(inc) / (2 * np.abs(bperp))
    b_crit = lam * rng * bw * np.tan(inc) / C                       # flat terrain
    b_crit_slope = lam * rng * bw * np.tan(np.radians(10.0)) / C    # slope facing the radar
    return dict(height=z[k], n=int(m.sum()), lam=lam, bw=bw,
                bperp=bperp, bpar=bpar, inc=np.degrees(inc), rng=rng, h_amb=h_amb,
                g_flat=1 - np.abs(bperp) / b_crit, g_slope=1 - np.abs(bperp) / b_crit_slope,
                b_crit=b_crit)


def fmt(a, d=1):
    """median (min to max), ignoring the cube's fill cells."""
    return f"{np.nanmedian(a):.{d}f} ({np.nanmin(a):.{d}f} to {np.nanmax(a):.{d}f})"


def main() -> None:
    for label, ref in PAIRS.items():
        gunw = next(g for g in GUNWS if ref in g.name.split("_")[11])
        r = baselines(gunw)
        print(f"\n=== {label}  {gunw.name} ===")
        print(f"  cube height {r['height']:.0f} m, {r['n']} cube cells over the ASO AOI "
              f"({np.isfinite(r['bperp']).mean() * 100:.0f}% with a baseline value), "
              f"lambda {r['lam']*100:.2f} cm, range bandwidth {r['bw']/1e6:.0f} MHz")
        print(f"  perpendicular baseline (m): {fmt(r['bperp'])}")
        print(f"  parallel baseline (m):      {fmt(r['bpar'])}")
        print(f"  incidence angle (deg):      {fmt(r['inc'])}")
        print(f"  slant range (km):           {fmt(r['rng']/1e3)}")
        print(f"  height of ambiguity (m):    {fmt(r['h_amb'], 0)}")
        print(f"  critical baseline, flat (km): {fmt(r['b_crit']/1e3)}")
        print(f"  geometric coherence, flat terrain:      {fmt(r['g_flat'], 4)}")
        print(f"  geometric coherence, 10 deg local inc.: {fmt(r['g_slope'], 3)}")


if __name__ == "__main__":
    main()
