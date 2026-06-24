"""Fetch the two NISAR GUNWs the paper's coherence analysis uses.

The multi-resolution coherence figures (corrected_coherence_*, coherence_rows)
are computed by nisar/calibrate_enl.py and nisar/debias_coherences.py, which
read two delivered standard GUNWs (descending path 042 over Tuolumne):

    GUNW_P12 : 2025-12-07 -> 2025-12-19
    GUNW_P23 : 2025-12-19 -> 2025-12-31

This downloads only those two (by their original ASF product names, no
renaming) into DATA_ROOT/sub_pixel_variability/raw/nisar/gunw/. Existing files
are skipped.

Requires Earthdata Login credentials in ~/.netrc and the nisar_pytools package.
Run with the nisar_pytools conda env:
    conda activate nisar_pytools
    python fetch_paper_gunws.py
"""

import sys
from pathlib import Path

from nisar_pytools import download_urls, find_nisar

# repo-relative data root (code/nisar/download/ -> code/ is two levels up)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tc_paths import DATA_ROOT  # noqa: E402

# Tuolumne ASO lidar bounds (EPSG:4326), same AOI as the other download scripts.
AOI = [-120.288, 37.714, -119.191, 38.241]   # [W, S, E, N]
PATH = 42                                     # descending track 042
START, END = "2025-12-05", "2026-01-02"       # brackets the Dec 07/19/31 scenes

# the two pairs we need, identified by the (reference, secondary) date tokens
# that NISAR GUNW filenames embed as YYYYMMDD
WANTED_PAIRS = [("20251207", "20251219"),     # P12
                ("20251219", "20251231")]     # P23

GUNW_DIR = DATA_ROOT / "sub_pixel_variability/raw/nisar/gunw"


def main() -> None:
    GUNW_DIR.mkdir(parents=True, exist_ok=True)

    # query ASF for GUNWs in the narrow window on the descending track
    urls = find_nisar(AOI, start_date=START, end_date=END,
                      product_type="GUNW", path_number=PATH)
    print(f"found {len(urls)} GUNW URLs on path {PATH} in {START}..{END}")

    # keep only the URLs whose filename contains both date tokens of a wanted pair
    selected = []
    for u in urls:
        name = u.rsplit("/", 1)[-1]
        if any(ref in name and sec in name for ref, sec in WANTED_PAIRS):
            selected.append(u)
            print("  selected", name)

    if len(selected) != len(WANTED_PAIRS):
        print(f"\nERROR: expected {len(WANTED_PAIRS)} GUNWs, matched {len(selected)}.")
        print("URLs returned:")
        for u in urls:
            print("   ", u.rsplit("/", 1)[-1])
        sys.exit(1)

    # skip any already on disk; download the rest (original names preserved)
    todo = [u for u in selected if not (GUNW_DIR / u.rsplit("/", 1)[-1]).exists()]
    if not todo:
        print("both GUNWs already present in", GUNW_DIR)
        return

    print(f"\ndownloading {len(todo)} file(s) to: {GUNW_DIR}")
    paths = download_urls(todo, GUNW_DIR, max_workers=2)
    for p in paths:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    print("\ndone -- GUNWs under", GUNW_DIR)


if __name__ == "__main__":
    main()
