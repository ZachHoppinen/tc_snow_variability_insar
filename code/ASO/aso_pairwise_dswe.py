"""ASO pairwise differencing -> per-pair dSWE maps (v4, publication).

One self-contained job: given two ASO flights of a basin, turn the pair into a
3 m Delta-SWE map. The recipe is exactly what the manuscript describes:

  1. difference the two snow-depth rasters                 (Delta-depth)
  2. calibrate one basin-constant density from the pair    (rho_event)
       rho_event = mean over valid pixels of dSWE / Delta-depth, from the
       50 m SWE and 50 m depth products (the only resolution ASO delivers SWE)
  3. convert the 3 m Delta-depth to Delta-SWE with that density
       dSWE = Delta-depth * rho_event / 1000          [m water equivalent]

Why a calibrated density instead of ASO's own 50 m dSWE? Because we need dSWE
at the native 3 m depth posting (for the sub-pixel coherence/bias analysis),
and ASO only models SWE at 50 m. A single per-pair density carries the 50 m
SWE information down onto the 3 m depth field.

Snow mask: a pixel is kept only if at least one flight has >= MIN_DEPTH_M of
snow at its 50 m parent cell. Below that, depth is at ASO's noise floor.

This module owns its own file I/O and math (no dependency on the v1/v2
pipeline). Run it to process all four Tuolumne pairs, write each 3 m dSWE map
to NetCDF, and emit a per-pair diagnostic figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.enums import Resampling

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo code/ dir
from tc_paths import DATA_ROOT, FIG_ROOT  # noqa: E402

# --- paths ------------------------------------------------------------------
# Per-pair flight rasters laid out by prep_aso.py:
#   PROCESSED_DIR/{date_a}_{date_b}/dswe/*_swe_50m.tif
#   PROCESSED_DIR/{date_a}_{date_b}/dsd/*_snowdepth_{3,50}m.tif
PROCESSED_DIR = DATA_ROOT / "sub_pixel_variability/processed/aso"
FIG_DIR = FIG_ROOT / "etc"

# product suffix -> the prep_aso subdir it lives in
SUBDIR = {"swe_50m.tif": "dswe",
          "snowdepth_3m.tif": "dsd",
          "snowdepth_50m.tif": "dsd"}

BASIN = "Tuolumne"

# The four Tuolumne accumulation pairs (water year, early flight, late flight).
PAIRS = [
    ("WY2023", "2023Mar02-03",    "2023Mar16-17"),
    ("WY2024", "2024Jan29",       "2024Feb27-28"),
    ("WY2025", "2025Feb08-09",    "2025Feb25"),
    ("WY2026", "2026Jan31-Feb01", "2026Feb27-28"),
]

# --- physical / filtering constants -----------------------------------------
MIN_DEPTH_M = 0.10          # snow-mask threshold (>= 10 cm at the 50 m cell)
MIN_DDEPTH_M = 0.10         # density: only |Delta-depth| >= 10 cm (avoid 0/0)
RHO_MIN, RHO_MAX = 50.0, 600.0   # density: physical accumulation-snow bounds


# --- file I/O ---------------------------------------------------------------
def pair_dir(date_a: str, date_b: str) -> Path:
    """The prep_aso per-pair folder for this pair."""
    return PROCESSED_DIR / f"{date_a}_{date_b}"


def find_tif(pdir: Path, date_token: str, suffix: str) -> Path:
    """One flight's product TIFF inside a pair folder (prep_aso layout)."""
    path = pdir / SUBDIR[suffix] / f"ASO_{BASIN}_{date_token}_{suffix}"
    assert path.exists(), f"missing {path}"
    return path


def open_raster(path: Path) -> xr.DataArray:
    """Open a single-band raster as a 2D DataArray with NaN nodata.

    Reads fully into memory and closes the file handle (via the context
    manager) so no dangling GDAL handle survives to interpreter shutdown --
    that handle is what produced the cosmetic "Error in sys.excepthook" noise.
    """
    with rioxarray.open_rasterio(path, masked=True) as da:
        da = da.squeeze(drop=True).load()
    return da.where(np.isfinite(da))


def assert_same_grid(a: xr.DataArray, b: xr.DataArray) -> None:
    """Fail loudly if two rasters are not on the identical grid.

    Every ASO flight for a basin is delivered on one common grid (verified:
    all Tuolumne flights share the 3 m and 50 m postings exactly), so pair
    differencing is a direct array subtraction with NO resampling. This guard
    makes that assumption explicit -- if a future product ever breaks it we
    stop here instead of silently bilinear-interpolating (which would smear
    the very sub-pixel structure this analysis is measuring).
    """
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    assert a.rio.transform().almost_equals(b.rio.transform()), (
        f"grid mismatch:\n  {a.rio.transform()}\n  {b.rio.transform()}")


# --- step 2: per-pair density from the 50 m products ------------------------
def pair_event_density(date_a: str, date_b: str) -> tuple[float, np.ndarray]:
    """Basin event density (kg/m^3) and the per-pixel density field at 50 m.

    rho = (dSWE / Delta-depth) * 1000 per pixel, then averaged over pixels
    that pass the accumulation/physical filter. Returns (mean_rho, rho_field)
    where rho_field is NaN outside the valid set (kept for plotting).
    """
    # ASO delivers every flight on one common 50 m grid, so this is a direct
    # difference -- no resampling. Guard the assumption, then subtract.
    pdir = pair_dir(date_a, date_b)
    sd_a = open_raster(find_tif(pdir, date_a, "snowdepth_50m.tif"))
    sd_b = open_raster(find_tif(pdir, date_b, "snowdepth_50m.tif"))
    swe_a = open_raster(find_tif(pdir, date_a, "swe_50m.tif"))
    swe_b = open_raster(find_tif(pdir, date_b, "swe_50m.tif"))
    for other in (sd_b, swe_a, swe_b):
        assert_same_grid(sd_a, other)

    ddepth = sd_b.values - sd_a.values
    dswe = swe_b.values - swe_a.values
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (dswe / ddepth) * 1000.0

    # Keep only accumulation pixels with a physical, sign-consistent ratio.
    valid = (np.isfinite(rho)
             & (np.abs(ddepth) >= MIN_DDEPTH_M)
             & (np.sign(ddepth) == np.sign(dswe))
             & (rho >= RHO_MIN) & (rho <= RHO_MAX))
    rho_field = np.where(valid, rho, np.nan)
    return float(np.nanmean(rho_field)), rho_field


# --- steps 1 + 3: 3 m depth difference -> 3 m dSWE map ----------------------
def pair_dswe_3m(date_a: str, date_b: str, rho: float) -> xr.DataArray:
    """3 m Delta-SWE map for the pair, snow-masked, in ASO's CRS (UTM 11N).

    dSWE = (depth_B - depth_A) * rho / 1000, at the native 3 m posting; then
    masked to pixels whose 50 m parent has >= MIN_DEPTH_M of snow in either
    flight (nearest-neighbour upsampled so the snow boundary stays crisp).
    """
    # 1. difference the 3 m depths -> Delta-depth (common grid, direct subtract)
    pdir = pair_dir(date_a, date_b)
    sd_a = open_raster(find_tif(pdir, date_a, "snowdepth_3m.tif"))
    sd_b = open_raster(find_tif(pdir, date_b, "snowdepth_3m.tif"))
    assert_same_grid(sd_a, sd_b)
    ddepth = sd_b - sd_a

    # 3. constant-density conversion to Delta-SWE (m water equivalent)
    dswe = ddepth * (rho / 1000.0)
    dswe.rio.write_crs(sd_a.rio.crs, inplace=True)

    # snow mask from the 50 m depths (>= MIN_DEPTH_M in EITHER flight)
    sd50_a = open_raster(find_tif(pdir, date_a, "snowdepth_50m.tif"))
    sd50_b = open_raster(find_tif(pdir, date_b, "snowdepth_50m.tif"))
    assert_same_grid(sd50_a, sd50_b)
    snow_50 = ((sd50_a >= MIN_DEPTH_M) | (sd50_b >= MIN_DEPTH_M)).astype(np.float32)
    snow_50.rio.write_crs(sd50_a.rio.crs, inplace=True)
    # genuine 50 m -> 3 m upsample (nearest keeps the snow boundary crisp)
    snow_3m = snow_50.rio.reproject_match(dswe, resampling=Resampling.nearest)

    dswe = dswe.where(snow_3m.values > 0.5)
    dswe.name = "dswe"
    dswe.attrs.update(rho_event_kgm3=rho, date_a=date_a, date_b=date_b,
                      units="m water equivalent")
    return dswe


# --- visualization ----------------------------------------------------------
def plot_pair(wy: str, date_a: str, date_b: str, rho: float,
              rho_field: np.ndarray, dswe: xr.DataArray) -> Path:
    """Three-panel diagnostic: 50 m density field, its histogram, 3 m dSWE."""
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.6),
                                        constrained_layout=True)

    # panel 0: per-pixel 50 m density (where it passed the filter)
    im0 = ax0.imshow(rho_field, cmap="viridis", vmin=RHO_MIN, vmax=RHO_MAX)
    fig.colorbar(im0, ax=ax0, shrink=0.8, label=r"$\rho$ (kg m$^{-3}$)")
    ax0.set_title("per-pixel density (50 m)")
    ax0.set_xticks([]); ax0.set_yticks([])

    # panel 1: density distribution + the basin mean we actually use
    finite_rho = rho_field[np.isfinite(rho_field)]
    ax1.hist(finite_rho, bins=np.arange(RHO_MIN, RHO_MAX + 10, 10), color="0.7")
    ax1.axvline(rho, color="firebrick", lw=2, label=f"mean = {rho:.0f}")
    ax1.set_xlabel(r"$\rho = \Delta$SWE/$\Delta$depth (kg m$^{-3}$)")
    ax1.set_ylabel("50 m pixels")
    ax1.set_title("density distribution")
    ax1.legend()

    # panel 2: the resulting 3 m dSWE map (symmetric diverging scale)
    d = dswe.values
    vmax = np.nanpercentile(np.abs(d), 99)
    im2 = ax2.imshow(d * 100.0, cmap="RdBu", vmin=-vmax * 100, vmax=vmax * 100)
    fig.colorbar(im2, ax=ax2, shrink=0.8, label=r"$\Delta$SWE (cm w.e.)")
    ax2.set_title(f"3 m $\\Delta$SWE  (median "
                  f"{np.nanmedian(d) * 100:.1f} cm)")
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle(f"{wy}:  {date_a} -> {date_b}   "
                 f"(rho_event = {rho:.0f} kg m$^{{-3}}$)", fontsize=14)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"aso_dswe_{wy}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    print(f"{'pair':24s} {'rho_event':>10s} {'n_snow_3m':>12s} "
          f"{'dSWE med (cm)':>14s}")
    for wy, date_a, date_b in PAIRS:
        print(f"\n[{wy}] {date_a} -> {date_b}")

        # step 2: one density for the pair
        rho, rho_field = pair_event_density(date_a, date_b)
        print(f"  rho_event = {rho:.1f} kg/m^3 "
              f"(from {np.isfinite(rho_field).sum():,} valid 50 m pixels)")

        # steps 1 + 3: the 3 m dSWE map
        dswe = pair_dswe_3m(date_a, date_b, rho)
        n_snow = int(np.isfinite(dswe.values).sum())
        med_cm = float(np.nanmedian(dswe.values)) * 100.0
        print(f"  3 m dSWE: {n_snow:,} snow pixels, median {med_cm:.2f} cm w.e.")

        # save the 3 m dSWE product into this pair's folder, next to its inputs
        nc_path = pair_dir(date_a, date_b) / "aso_dswe_3m.nc"
        dswe.to_netcdf(nc_path)
        fig_path = plot_pair(wy, date_a, date_b, rho, rho_field, dswe)
        print(f"  wrote {nc_path} and {fig_path.name}")

        print(f"{wy + ' ' + date_a + '->' + date_b:24s} {rho:10.0f} "
              f"{n_snow:12,d} {med_cm:14.2f}")


if __name__ == "__main__":
    main()
