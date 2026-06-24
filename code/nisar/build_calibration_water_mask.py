"""Build and save the canonical Don Pedro open-water pixel raster used for BOTH
the NISAR and S1 oversample/Bamler calibrations, so the two rest on the exact
same water cells.

Selection (the same one the NISAR water calibration uses):
    water = WorldCover class 80 (permanent water)  AND  NISAR gamma_80 < 0.20
The coh<0.20 cut drops shoreline-mixed pixels and the frozen/coherent alpine
reservoirs, leaving the truly decorrelated Don Pedro open water.

NISAR gamma_80 is the coherenceMagnitude from the low-res (15x20 crossmul, 80 m)
two-stage GUNW. Saved to cache/calibration_water_pixels_80m.tif (1 = calibration
water, 0 = not) on the 80 m grid with CRS. Reproject-match this onto any sensor
grid to sample the identical water cells.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

DATA = Path("/Users/zmhoppinen/Documents/nisar_swe/data")
CACHE = Path("/Users/zmhoppinen/Documents/nisar_swe/projects/"
             "theoretical_sweramp_unwrapping_coherence/cache")
# all v4 outputs live under the sub_pixel_variability data tree
OUT_DIR = DATA / "sub_pixel_variability/processed/calibration"

GUNW_LOW = DATA / "nisar/gunw/multi_resolution_tuolome/lowres_crossmul_15x20/product.h5"
WORLDCOVER_WATER_MASK = CACHE / "tuolome_water_mask_worldcover_80m.tif"   # v3 input
OUT = OUT_DIR / "calibration_water_pixels_80m.tif"

WATER_COH_MAX = 0.20                                  # open-water coherence cutoff
DONPEDRO_BBOX_UTM = (770000, 4204000, 790000, 4215000)   # EPSG:32610, display ref


def read_nisar_coh(gunw_path: Path) -> xr.DataArray:
    """NISAR 80 m coherenceMagnitude from a two-stage GUNW as a CRS-aware grid."""
    base = "science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH"
    with h5py.File(gunw_path, "r") as f:
        coh = f[f"{base}/coherenceMagnitude"][:].astype(np.float32)
        x = f[f"{base}/xCoordinates"][:]
        y = f[f"{base}/yCoordinates"][:]
        epsg = int(f[f"{base}/projection"][()])
    da = xr.DataArray(coh, dims=("y", "x"), coords={"y": y, "x": x}, name="coherence")
    da.rio.write_crs(f"EPSG:{epsg}", inplace=True)
    return da


def main():
    low = read_nisar_coh(GUNW_LOW)                    # NISAR 80 m coherence
    water = rioxarray.open_rasterio(WORLDCOVER_WATER_MASK).squeeze(drop=True)
    water_on = water.rio.reproject_match(low, resampling=Resampling.nearest)

    wc = water_on.values == 1
    coh = low.values
    mask = wc & np.isfinite(coh) & (coh < WATER_COH_MAX)

    out = xr.DataArray(mask.astype("uint8"), dims=low.dims, coords=low.coords,
                       name="calibration_water")
    out.rio.write_crs(low.rio.crs, inplace=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.rio.to_raster(OUT)

    x0, y0, x1, y1 = DONPEDRO_BBOX_UTM
    X, Y = np.meshgrid(low.x.values, low.y.values)
    indp = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    print(f"WorldCover class-80 pixels      : {int(wc.sum()):,}")
    print(f"  AND finite NISAR gamma_80       : {int((wc & np.isfinite(coh)).sum()):,}")
    print(f"  AND gamma_80 < {WATER_COH_MAX}          : {int(mask.sum()):,}  (canonical water)")
    print(f"  of those, inside Don Pedro bbox : {int((mask & indp).sum()):,}")
    print(f"  median NISAR gamma_80 over water : {np.nanmedian(coh[mask]):.4f}")
    print(f"CRS {out.rio.crs}  ->  wrote {OUT}")


if __name__ == "__main__":
    main()
