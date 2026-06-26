"""Fetch VIIRS VNP10A1F snow cover for the two NISAR pair windows (Appendix B).

Downloads the VIIRS/NPP CGF Snow Cover Daily L3 375 m product (VNP10A1F,
Version 2; Riggs & Hall 2022, NSIDC DAAC, doi:10.5067/PN50Y51IVNLE) for three
dates spanning each NISAR pair's temporal baseline:

    P12 (no accumulation) : 2025-12-07, 2025-12-13, 2025-12-19
    P23 (large storm)     : 2025-12-19, 2025-12-25, 2025-12-31

Each granule (sinusoidal tile h08v05, which covers Tuolumne) is written as a
CRS-tagged GeoTIFF to DATA_ROOT/sub_pixel_variability/raw/viirs/ as
viirs_<pair>_<date>.tif. Existing files are skipped.

Requires Earthdata Login credentials in ~/.netrc and the `earthaccess` package.
Run, e.g.:
    conda activate nisar_pytools      # (needs earthaccess installed)
    python fetch_viirs_snowcover.py
"""

import sys
from pathlib import Path

import earthaccess
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

# repo-relative data root (code/viirs/ -> code/ is one level up)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tc_paths import DATA_ROOT  # noqa: E402

AOI_LL = [-120.288, 37.714, -119.191, 38.241]   # [W, S, E, N], Tuolumne ASO bounds
SHORT_NAME = "VNP10A1F"
# three dates per pair spanning its temporal baseline (must match the windows in
# stats/viirs_snowcover_stats.py)
PAIR_WINDOWS = {"p12": ["2025-12-07", "2025-12-13", "2025-12-19"],
                "p23": ["2025-12-19", "2025-12-25", "2025-12-31"]}
VIIRS_DIR = DATA_ROOT / "sub_pixel_variability/raw/viirs"

# VIIRS sinusoidal grid geometry for tile h08v05 (the one covering Tuolumne).
_TILE_M = 1111950.519667
_PS = _TILE_M / 3000.0                       # 370.65 m / pixel (3000x3000 tile)
_TRANSFORM = Affine(_PS, 0.0, -11119505.196667,    # x0 at h = 8
                    0.0, -_PS, 10007554.676667 - 5 * _TILE_M)  # y0 at v = 5
_SINU = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
                       "+R=6371007.181 +units=m +no_defs")


def fetch_one(pair: str, date: str) -> None:
    """Download one VNP10A1F day and write the NDSI snow-cover band as a tif."""
    out = VIIRS_DIR / f"viirs_{pair}_{date}.tif"
    if out.exists():
        print(f"  cached: {out.name}")
        return
    results = earthaccess.search_data(short_name=SHORT_NAME,
                                      bounding_box=tuple(AOI_LL),
                                      temporal=(date, date))
    if not results:
        print(f"  {pair} {date}: no granules found")
        return
    files = earthaccess.download(results, str(VIIRS_DIR))
    h5 = next((str(f) for f in files if str(f).endswith((".h5", ".hdf"))), None)
    if h5 is None:
        print(f"  {pair} {date}: no HDF5 in download")
        return

    # pull the NDSI_Snow_Cover subdataset (skip the QA layer of the same name)
    with rasterio.open(h5) as src:
        sub = next(s for s in src.subdatasets
                   if "NDSI_Snow_Cover" in s and "Basic_QA" not in s)
    with rasterio.open(sub) as src:
        arr = src.read(1)

    with rasterio.open(out, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype=arr.dtype,
                       crs=_SINU, transform=_TRANSFORM, nodata=255) as dst:
        dst.write(arr, 1)
    print(f"  wrote {out.name}")


def main() -> None:
    VIIRS_DIR.mkdir(parents=True, exist_ok=True)
    print("authenticating with earthaccess ...")
    earthaccess.login(strategy="netrc")
    for pair, dates in PAIR_WINDOWS.items():
        print(f"--- {pair} ---")
        for date in dates:
            fetch_one(pair, date)
    print("\ndone -- VIIRS snow cover under", VIIRS_DIR)


if __name__ == "__main__":
    main()
