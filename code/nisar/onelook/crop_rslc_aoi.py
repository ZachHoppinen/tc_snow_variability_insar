"""Crop the local Tuolumne RSLCs to a small box around a 1 km AOI so a
single-look interferogram can be formed quickly (referee 1 explainer figure).

Uses hyp3_isce3.crop_rslc on the full RSLC files already on disk: solve the
radar window for the box with aoi_to_radar_window (1 x 1 looks, so no look
snapping), then copy only that window to <date>_sub.h5. Every date is cropped
to the same box so a pair shares one geometry.

Output (local, under the repo): data/processed/nisar/onelook/<aoi>/crops/
    <YYYYMMDD>_sub.h5   and a copy of the DEM as dem.tif

Run in the hyp3-isce3 env from the repo root:
    conda run -n hyp3-isce3 python code/nisar/onelook/crop_rslc_aoi.py aoi3 20251219 20251231
"""

from __future__ import annotations

import argparse
import shutil

from hyp3_isce3.crop_rslc import aoi_to_radar_window, crop_rslc, get_polarizations
from nisar.products.readers import SLC
from pyproj import Transformer

from onelook_config import AOIS, DEM_SRC, HALF, MARGIN, OUT_ROOT, RSLC_DIR  # noqa: E402


def bbox_wgs84(center: tuple[float, float]) -> list[float]:
    tr = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(center[0] - HALF, center[1] - HALF)
    lon1, lat1 = tr.transform(center[0] + HALF, center[1] + HALF)
    return [lon0, lat0, lon1, lat1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("aoi", choices=AOIS)
    ap.add_argument("dates", nargs="+", help="YYYYMMDD of each RSLC to crop")
    a = ap.parse_args()

    out = OUT_ROOT / a.aoi / "crops"
    out.mkdir(parents=True, exist_ok=True)
    dem = out / "dem.tif"
    if not dem.exists():
        shutil.copy(DEM_SRC, dem)
    bbox = bbox_wgs84(AOIS[a.aoi])
    print("crop box WGS84 [W, S, E, N]:", [round(v, 4) for v in bbox], flush=True)

    for d in a.dates:
        dst = out / f"{d}_sub.h5"
        if dst.exists():
            print(f"{d}: have {dst.name}", flush=True)
            continue
        srcs = sorted(RSLC_DIR.glob(f"NISAR_L1_PR_RSLC_*_{d}T*.h5"))
        if len(srcs) != 1:
            raise SystemExit(f"{d}: expected one RSLC, found {srcs}")
        slc = SLC(hdf5file=str(srcs[0]))
        window = aoi_to_radar_window(slc, bbox, dem, margin=MARGIN, az_looks=1, rg_looks=1)
        print(f"{d}: window {window}", flush=True)
        crop_rslc(srcs[0], dst, window, get_polarizations(slc))
        print(f"{d}: wrote {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
