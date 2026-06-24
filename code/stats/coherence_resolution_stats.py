"""coherence_resolution_stats.py -- multi-resolution NISAR coherence stats,
computed directly on the native coherence files (no reprojection).

The 80 m products are an exact 4x4 block aggregation of the 20 m products
(identical origin and CRS, sizes differ by 4), so the 20 m coherence is brought
onto the 80 m grid by a plain 4x4 block mean -- no warping or resampling of the
coherence values. The corrected layers are the Bamler-debiased coherences from
debias_coherences.py.

Per pair (P12 quiet, P23 storm) we report, over the full finite footprint:
  - raw and corrected coherence at 20 m and 80 m (mean, median, IQR)
  - raw gap (gamma_20 - gamma_80) and corrected dgamma = gamma_tilde_20 -
    gamma_tilde_80 on the 80 m grid (mean, median, IQR, % positive)
dgamma is the NISAR estimate of sub-pixel-variability coherence loss: ~0 for
the quiet pair, positive for the storm pair.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

warnings.filterwarnings("ignore", category=RuntimeWarning)

COH_DIR = Path("/Users/zmhoppinen/Documents/nisar_swe/data/"
               "sub_pixel_variability/processed/nisar/coherence")
PROCESSED_DIR = Path("/Users/zmhoppinen/Documents/nisar_swe/data/"
                     "sub_pixel_variability/processed/aso")
DEM_TIF = Path("/Users/zmhoppinen/Documents/nisar_swe/data/"
               "sub_pixel_variability/raw/nisar/dem.tif")
SNOW_PAIR = ("2025Feb08-09", "2025Feb25")     # ASO footprint used as snow mask
PAIRS = [("P12", "7-19 Dec, low snow"), ("P23", "19-31 Dec, large storm")]
F = 4                                    # 20 m -> 80 m block factor


def read(name: str) -> np.ndarray:
    with rioxarray.open_rasterio(COH_DIR / name, masked=True) as da:
        return da.squeeze(drop=True).values


def read_da(name: str) -> xr.DataArray:
    with rioxarray.open_rasterio(COH_DIR / name, masked=True) as da:
        return da.squeeze(drop=True).load()


def snow_mask_on(template: xr.DataArray) -> np.ndarray:
    """ASO snow footprint reprojected (nearest) onto the NISAR 80 m grid.
    Only the binary mask is reprojected; the coherence is never resampled."""
    pdir = PROCESSED_DIR / f"{SNOW_PAIR[0]}_{SNOW_PAIR[1]}"
    with xr.open_dataset(pdir / "aso_M_81m_L.nc") as ds:
        fp = np.isfinite(ds["M_real"].values).astype(np.float32)
        da = xr.DataArray(fp, dims=ds["M_real"].dims, coords=ds["M_real"].coords)
    da.rio.write_crs("EPSG:32611", inplace=True)
    return da.rio.reproject_match(template, resampling=Resampling.nearest).values > 0.5


def dem_on(template: xr.DataArray) -> np.ndarray:
    """Elevation (m) reprojected onto the NISAR 80 m grid (DEM resampled, not
    the coherence)."""
    with rioxarray.open_rasterio(DEM_TIF, masked=True) as d0:
        dem = d0.squeeze(drop=True).load()
    return dem.rio.reproject_match(template, resampling=Resampling.bilinear).values


def coarsen4(a: np.ndarray) -> np.ndarray:
    """NaN-safe 4x4 block mean: 20 m grid -> the (aligned) 80 m grid."""
    ny, nx = (a.shape[0] // F) * F, (a.shape[1] // F) * F
    b = a[:ny, :nx].reshape(ny // F, F, nx // F, F)
    return np.nanmean(b, axis=(1, 3))


def summ(name: str, a: np.ndarray) -> str:
    a = a[np.isfinite(a)]
    p25, p50, p75 = np.percentile(a, [25, 50, 75])
    return (f"{name:20s} mean={a.mean():+.3f} median={p50:+.3f} "
            f"IQR=[{p25:+.3f},{p75:+.3f}] (n={a.size:,})")


def main() -> None:
    for prefix, label in PAIRS:
        c80_da = read_da(f"{prefix}_coherence_corrected_80m.tif")  # 80 m template
        snow = snow_mask_on(c80_da)                                # snow on 80 m grid
        r20 = read(f"{prefix}_coherence_raw_20m.tif")
        r80 = read(f"{prefix}_coherence_raw_80m.tif")
        c20 = read(f"{prefix}_coherence_corrected_20m.tif")
        c80 = c80_da.values

        raw_gap = coarsen4(r20) - r80          # both on the native 80 m grid
        corr_gap = coarsen4(c20) - c80

        print(f"\n=== {prefix} ({label}) ===")
        print("  " + summ("raw  gamma_20 (20m)", r20))
        print("  " + summ("raw  gamma_80 (80m)", r80))
        print("  " + summ("corr gamma_20 (20m)", c20))
        print("  " + summ("corr gamma_80 (80m)", c80))
        for tag, mask in [("FULL scene", np.ones_like(corr_gap, bool)),
                          ("SNOW only ", snow)]:
            rg, cgp = raw_gap[mask], corr_gap[mask]
            cgf = cgp[np.isfinite(cgp)]
            print(f"  [{tag}]  raw gap: median={np.nanmedian(rg):+.3f}  "
                  f"corr dgamma: mean={np.nanmean(cgp):+.3f} "
                  f"median={np.nanmedian(cgp):+.3f} "
                  f"IQR=[{np.nanpercentile(cgp,25):+.3f},"
                  f"{np.nanpercentile(cgp,75):+.3f}]  "
                  f">0: {100*np.mean(cgf>0):.0f}%  (n={cgf.size:,})")

        # corrected dgamma vs elevation (snow only), to show the low->high trend
        elev = dem_on(c80_da)
        m = snow & np.isfinite(corr_gap) & np.isfinite(elev)
        e, cg, g80 = elev[m], corr_gap[m], r80[m]   # raw 80 m coherence for context
        edges = np.percentile(e, [0, 20, 40, 60, 80, 100])
        print("  [SNOW] corr dgamma by elevation quintile "
              "(median / p75 / p90):")
        for i in range(5):
            lo, hi = edges[i], edges[i + 1]
            sel = (e >= lo) & (e <= hi if i == 4 else e < hi)
            d = cg[sel]
            print(f"    {lo:6.0f}-{hi:6.0f} m  dgamma med={np.median(d):+.3f} "
                  f"p75={np.percentile(d,75):+.3f} p90={np.percentile(d,90):+.3f}  "
                  f"raw_g80={np.median(g80[sel]):.3f}  (n={sel.sum():,})")


if __name__ == "__main__":
    main()
