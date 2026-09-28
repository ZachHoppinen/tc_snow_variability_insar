"""Study area map (referee 1, Figure 4 terrain comment).

One map on the 81 m M grid used by the |M| figures: DEM elevation tint under a
soft hillshade, faded outside the ASO lidar footprint, the footprint outline,
the four labeled CDEC snow pillows, a California locator inset, and a NISAR acquisition
geometry inset (flight and imaging directions from the GUNW metadata).

Inputs (all already used elsewhere in the pipeline):
  DEM         DATA_ROOT/sub_pixel_variability/raw/nisar/dem.tif   (WGS84, 1 arcsec)
  footprint   the WY2023 aso_M_81m_L.nc grid (EPSG:32611), finite |M| cells
  geometry    the P23 GUNW radarGrid unit vectors
  water       the ENL calibration water pixels, selected exactly as in
              code/nisar/calibrate_enl.py (WorldCover class 80, gamma_80 < 0.20, per
              pair), union over the P12 and P23 GUNWs
Pillow coordinates are the CDEC station metadata (staMeta pages), copied here
so the figure needs no network access.

Run with TC_DATA_ROOT pointing at the data tree:
    conda run -n nisar_pytools python code/visualizations/plot_study_area.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cartopy.io.shapereader as shpreader
import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap, Normalize
import numpy as np
import rioxarray
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy import ndimage

from plotting_constants import apply_style, DATA_ROOT, FIG_DIR, PROCESSED_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nisar"))
import calibrate_enl  # noqa: E402  (water selection: read_coh, WATER_MASK, COH_MAX)

DEM = DATA_ROOT / "sub_pixel_variability/raw/nisar/dem.tif"
GUNW_DIR = DATA_ROOT / "sub_pixel_variability/raw/nisar/gunw"
GUNW = GUNW_DIR / "GUNW_P23_20251219_20251231.h5"
CAL_GUNWS = [GUNW_DIR / "GUNW_P12_20251207_20251219.h5", GUNW]    # the two ENL calibration pairs
M_GRID = PROCESSED_DIR / "2023Mar02-03_2023Mar16-17/aso_M_81m_L.nc"
CRS = "EPSG:32611"
# muted hypsometric ramp: valley green -> forest tan -> alpine rock -> near white
ELEV_CMAP = LinearSegmentedColormap.from_list(
    "elev", ["#7d9a6a", "#b5bf8a", "#d8cfa0", "#bfa98a", "#a8a39c", "#f2f0ec"])
ELEV_NORM = Normalize(800, 4000)
WATER_COLOR = "#1f5fa8"

# CDEC snow pillows: code, name, lat, lon (CDEC station metadata, Sept 2026)
PILLOWS = [("DAN", "Dana Meadows", 37.898045, -119.258808),
           ("TUM", "Tuolumne Meadows", 37.876215, -119.349968),
           ("SLI", "Slide Canyon", 38.091234, -119.431881),
           ("GIN", "Gin Flat", 37.766887, -119.774907)]


def footprint_template():
    """The 81 m M grid as a template DataArray, plus the finite-|M| footprint."""
    with xr.open_dataset(M_GRID) as ds:
        m = ds["M_real"].load()
    tmpl = xr.DataArray(np.zeros(m.shape, "float32"), dims=m.dims, coords=m.coords)
    tmpl.rio.write_crs(CRS, inplace=True)
    return tmpl, np.isfinite(m.values)


def on_grid(path, tmpl, resampling):
    """Reproject a raster onto the M grid."""
    with rioxarray.open_rasterio(path, masked=True) as r:
        return r.squeeze(drop=True).rio.reproject_match(tmpl, resampling=resampling).values


def calibration_water(tmpl):
    """ENL calibration water on the M grid: per pair, WorldCover water with finite 20 and
    80 m coherence and gamma_80 < COH_MAX (as in calibrate_enl.water_pairs), OR'd over pairs."""
    wc = rioxarray.open_rasterio(calibrate_enl.WATER_MASK).squeeze("band", drop=True)
    water = np.zeros(tmpl.shape, bool)
    n_total = 0
    for gunw in CAL_GUNWS:
        c80 = calibrate_enl.read_coh(gunw, "unwrappedInterferogram")
        c20 = calibrate_enl.read_coh(gunw, "wrappedInterferogram").rio.reproject_match(
            c80, resampling=Resampling.average)
        m = wc.rio.reproject_match(c80, resampling=Resampling.nearest).values == 1
        sel = m & np.isfinite(c80.values) & np.isfinite(c20.values) & (c80.values < calibrate_enl.COH_MAX)
        n_total += int(sel.sum())
        sel_da = xr.DataArray(sel.astype("uint8"), dims=c80.dims, coords=c80.coords).rio.write_crs(c80.rio.crs)
        water |= sel_da.rio.reproject_match(tmpl, resampling=Resampling.nearest).values == 1
    print(f"calibration water: {n_total} samples over both pairs at 80 m")
    return water


def sar_geometry():
    """Flight and imaging azimuths (deg clockwise from north) and incidence
    from the GUNW radar grid unit vectors. losUnitVector points ground -> sensor."""
    with h5py.File(GUNW) as h:
        g = h["science/LSAR/GUNW/metadata/radarGrid"]
        med = lambda k: float(np.nanmedian(g[k][()]))
        ax_, ay_ = med("alongTrackUnitVectorX"), med("alongTrackUnitVectorY")
        lx, ly = med("losUnitVectorX"), med("losUnitVectorY")
        inc = med("incidenceAngle")
    flight_az = np.degrees(np.arctan2(ax_, ay_)) % 360
    imaging_az = (np.degrees(np.arctan2(lx, ly)) + 180) % 360      # sensor -> ground
    return flight_az, imaging_az, inc


def california_locator(ax, site_ll):
    """California state outline (Natural Earth) with a red dot on the basin."""
    shapefile = shpreader.natural_earth(resolution="50m", category="cultural",
                                        name="admin_1_states_provinces")
    for record in shpreader.Reader(shapefile).records():
        if record.attributes.get("name") != "California":
            continue
        geom = record.geometry
        for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            px, py = poly.exterior.xy
            ax.fill(px, py, fc="0.82", ec="0.5", lw=0.8, alpha=0.9)
    ax.plot(*site_ll, "o", color="red", mec="k", mew=0.6, ms=7, zorder=5)
    ax.annotate("Tuolumne", site_ll, textcoords="offset points", xytext=(8, -9),
                fontsize=7, color="red", fontweight="bold", ha="center", va="top")
    ax.set_aspect(1 / np.cos(np.radians(37)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def geometry_inset(ax, rect, track_label):
    """Acquisition geometry inset: flight track (black) with an arrowhead, imaging
    direction (blue) from the track, north arrow and track label. Same format as the
    nival_validation context figure, scaled to fill the box."""
    flight_az, imaging_az, inc = sar_geometry()
    ins = ax.inset_axes(rect)
    ins.set_xlim(-1.15, 1.15); ins.set_ylim(-1.45, 1.15); ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("#e9edf3"); ins.patch.set_alpha(0.92)
    for spine in ins.spines.values():
        spine.set_edgecolor("k"); spine.set_linewidth(1.0)
    d = lambda az: (np.sin(np.radians(az)), np.cos(np.radians(az)))
    ox = -0.35                                   # origin shifted left in the box
    fx, fy = d(flight_az)
    ins.plot([ox - fx * 0.95, ox + fx * 0.95], [-fy * 0.95, fy * 0.95], color="k", lw=2.2, zorder=3)
    ins.annotate("", xy=(ox + fx * 1.02, fy * 1.02), xytext=(ox + fx * 0.6, fy * 0.6),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=2.2, mutation_scale=13), zorder=3)
    lx, ly = d(imaging_az)
    ins.annotate("", xy=(ox + lx * 0.85, ly * 0.85), xytext=(ox, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=2.2, mutation_scale=13), zorder=4)
    ins.plot(ox, 0, "o", color="k", ms=3, zorder=5)
    # labels: flight at the tail end of the track, imaging beside the blue arrow tip
    ins.text(ox - fx * 0.75 + 0.55, -fy * 0.75 - 0.15, "flight\ndirection", fontsize=7.5,
             fontweight="bold", ha="center", va="center", linespacing=0.9)
    ins.text(ox + lx * 0.85 + 0.05, ly * 0.85 - 0.28, "imaging\ndirection", color="#1565c0", fontsize=7.5,
             fontweight="bold", ha="center", va="center", linespacing=0.9)
    ins.annotate("", xy=(-0.95, 1.0), xytext=(-0.95, 0.65),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, mutation_scale=8))
    ins.text(-0.78, 0.82, "N", fontsize=8, fontweight="bold", ha="left", va="center")
    ins.text(0.0, -1.28, track_label, fontsize=8, fontweight="bold", ha="center", va="center")
    return flight_az, imaging_az, inc


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    tmpl, snow = footprint_template()
    elev = on_grid(DEM, tmpl, Resampling.bilinear)
    x, y = tmpl.x.values, tmpl.y.values
    dx = float(x[1] - x[0])
    extent = (x[0] - dx / 2, x[-1] + dx / 2, y[-1] - dx / 2, y[0] + dx / 2)
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    # outer boundary only: fill the lake and no-snow holes, keep the largest region
    outer = ndimage.binary_fill_holes(ndimage.binary_closing(snow, iterations=3))
    lab, n = ndimage.label(outer)
    outer = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    # elevation tint under a soft hillshade
    filled = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    rgb = LightSource(azdeg=315, altdeg=45).shade_rgb(
        ELEV_CMAP(ELEV_NORM(filled))[..., :3], filled, vert_exag=2.75, dx=dx, dy=dx,
        blend_mode="soft", fraction=0.9)
    # 50 % white wash outside the lidar footprint so the domain stands out
    rgb = np.where(outer[..., None], rgb, 0.5 * rgb + 0.5)
    ax.imshow(np.clip(rgb, 0, 1), extent=extent, origin="upper", zorder=0)
    # calibration open-water pixels in solid blue, inside the lidar footprint only
    water = calibration_water(tmpl) & outer
    ax.imshow(np.ma.masked_where(~water, water), extent=extent, origin="upper",
              cmap=ListedColormap([WATER_COLOR]), interpolation="nearest", zorder=1)
    ax.contour(outer.astype(float), levels=[0.5], extent=extent, origin="upper",
               colors="k", linewidths=1.4)
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    for code, name, lat, lon in PILLOWS:
        px, py = to_utm.transform(lon, lat)
        ax.plot(px, py, "^", color="white", mec="k", mew=1.2, ms=12, zorder=6)
        ax.annotate(code, (px, py), xytext=(7, 4), textcoords="offset points", fontsize=10,
                    fontweight="bold", path_effects=halo, zorder=6)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
    km = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")        # plain km ticks, no 1e6 offset
    ax.xaxis.set_major_formatter(km); ax.yaxis.set_major_formatter(km)
    ax.set_xlabel("UTM 11N easting (km)"); ax.set_ylabel("northing (km)")
    cb = fig.colorbar(plt.cm.ScalarMappable(ELEV_NORM, ELEV_CMAP), ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("elevation (m)")
    ax.legend(handles=[
        Line2D([], [], color="k", lw=1.4, label="ASO lidar coverage"),
        Line2D([], [], marker="s", color="none", mfc=WATER_COLOR, mec="none", ms=10, label="calibration water"),
        Line2D([], [], marker="^", color="none", mfc="white", mec="k", ms=11, label="CDEC snow pillow")],
        loc="lower right", fontsize=11, framealpha=0.9)

    # California locator, upper left: the state shape alone, no box behind it
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    clon, clat = to_ll.transform(x.mean(), y.mean())
    california_locator(ax.inset_axes([0.77, 0.70, 0.26, 0.29]), (clon, clat))

    flight_az, imaging_az, inc = geometry_inset(ax, [-0.02, 0.015, 0.24, 0.30], "NISAR DESC 042")

    out = FIG_DIR / "3_study_area.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"footprint {snow.sum()} cells, elevation {np.nanmin(elev[snow]):.0f}-{np.nanmax(elev[snow]):.0f} m")
    print(f"calibration water inside the lidar footprint: {water.sum()} cells on the 81 m grid")
    print(f"SAR geometry: flight {flight_az:.0f}, imaging {imaging_az:.0f}, incidence {inc:.0f} deg")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
