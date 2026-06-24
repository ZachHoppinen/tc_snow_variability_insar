"""prep_aso.py - unzip raw ASO flight archives into the v4 per-pair layout.

The raw ASO products arrive as one "AllData_and_Reports" zip per flight. For
the sub-pixel-variability work we only need three rasters from each flight:
the 50 m SWE, the 3 m snow depth, and the 50 m snow depth. This script pulls
just those three out of each zip and lays them out per pair, grouped by
product type, exactly how aso_pairwise_dswe.py expects to read them:

    processed/aso/{date_a}_{date_b}/
        dswe/  ASO_<basin>_<date_a>_swe_50m.tif        (both flights)
               ASO_<basin>_<date_b>_swe_50m.tif
        dsd/   ASO_<basin>_<date_a>_snowdepth_3m.tif   (both flights)
               ASO_<basin>_<date_a>_snowdepth_50m.tif
               ASO_<basin>_<date_b>_snowdepth_3m.tif
               ASO_<basin>_<date_b>_snowdepth_50m.tif

No differencing here -- that lives downstream in aso_pairwise_dswe.py. We just
extract and reorganize. The 3 m depth tiffs are ~2.3 GB each and are kept on
disk after extraction; an already-present file is left untouched so re-runs
are cheap.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo code/ dir
from tc_paths import DATA_ROOT  # noqa: E402

BASIN = "Tuolumne"
RAW_DIR = DATA_ROOT / "sub_pixel_variability/raw/aso"
PROCESSED_DIR = DATA_ROOT / "sub_pixel_variability/processed/aso"

# (early flight, late flight) date tokens -- the four Tuolumne accumulation
# pairs, same tokens aso_pairwise_dswe.py uses.
PAIRS = [
    ("2023Mar02-03",    "2023Mar16-17"),
    ("2024Jan29",       "2024Feb27-28"),
    ("2025Feb08-09",    "2025Feb25"),
    ("2026Jan31-Feb01", "2026Feb27-28"),
]

# product suffix -> destination subdir within the pair folder
LAYOUT = {
    "swe_50m.tif":       "dswe",
    "snowdepth_3m.tif":  "dsd",
    "snowdepth_50m.tif": "dsd",
}

COPY_CHUNK = 16 * 1024 * 1024     # 16 MB streaming copy buffer


def zip_path(date: str) -> Path:
    """Path to one flight's raw archive."""
    p = RAW_DIR / f"ASO_{BASIN}_{date}_AllData_and_Reports.zip"
    assert p.exists(), f"missing raw zip: {p}"
    return p


def member_name(date: str, suffix: str) -> str:
    """The exact in-zip filename for a flight's product (no cloud-adj sibling)."""
    return f"ASO_{BASIN}_{date}_{suffix}"


def extract_flight(date: str, pair_dir: Path) -> list[Path]:
    """Extract one flight's three products into the pair folder; return paths.

    Streams each member straight to its destination subdir. Skips a member
    whose target already exists so re-runs don't re-extract 2.3 GB tiffs.
    """
    targets: list[Path] = []
    with zipfile.ZipFile(zip_path(date)) as zf:
        names = set(zf.namelist())
        for suffix, subdir in LAYOUT.items():
            name = member_name(date, suffix)
            assert name in names, f"{name} not found in {zip_path(date).name}"
            out_sub = pair_dir / subdir
            out_sub.mkdir(parents=True, exist_ok=True)
            target = out_sub / name
            if target.exists():
                print(f"      have {subdir}/{name}")
            else:
                print(f"      extracting {subdir}/{name} ...")
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=COPY_CHUNK)
            targets.append(target)
    return targets


def main() -> None:
    for date_a, date_b in PAIRS:
        pair_dir = PROCESSED_DIR / f"{date_a}_{date_b}"
        print(f"[{date_a}_{date_b}] -> {pair_dir}")
        for date in (date_a, date_b):
            print(f"  flight {date}")
            for target in extract_flight(date, pair_dir):
                # cheap sanity check: read the georeference header only
                with rasterio.open(target) as ds:
                    epsg = ds.crs.to_epsg() if ds.crs else "?"
                    print(f"        {target.relative_to(pair_dir)}: "
                          f"{ds.width}x{ds.height}, {ds.res[0]:.0f} m, EPSG:{epsg}")
    print("\ndone -- ready for aso_pairwise_dswe.py")


if __name__ == "__main__":
    main()
