"""Download the two NISAR RSLCs that form the 2025-12-31 -> 2026-01-12
interferometric pair on descending path 042 over the Tuolumne ASO AOI.

NOTE: these RSLCs are NOT needed to reproduce the paper -- its NISAR coherence
analysis uses two delivered GUNWs directly (see fetch_paper_gunws.py); no SAS
run on RSLCs is involved. This script is kept only to document how the RSLCs
used in exploratory closure tests were obtained.

Pulls from ASF via nisar_pytools and writes to
``DATA_ROOT/nisar/rslc/tuolome/``. Existing files are skipped.
Requires Earthdata Login credentials in ``~/.netrc``.

Run with the ``nisar_pytools`` conda env, e.g.:

    conda activate nisar_pytools
    python download_rslcs.py
"""

import sys
from pathlib import Path

from nisar_pytools import download_urls, find_nisar

# repo-relative data root (code/nisar/download/ -> code/ is two levels up)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tc_paths import DATA_ROOT  # noqa: E402

# Tuolumne ASO lidar bounds (EPSG:4326), same AOI as the GUNW download
# script so the RSLC frame fully covers the basin.
AOI = [-120.288, 37.714, -119.191, 38.241]   # [W, S, E, N]

# Search window that brackets the two acquisitions of interest. The
# GUNW filename next to this pair confirms the dates:
#   ..._20251231T025608_20251231T025643_20260112T025609_20260112T025644...
START = "2025-12-28"
END = "2026-01-15"
PATH = 42                                      # descending track 042
OUT_DIR = DATA_ROOT / "nisar/rslc/tuolome"


def main() -> None:
    # Step 1: query ASF for matching RSLC URLs in the search window.
    urls = find_nisar(
        AOI,
        start_date=START,
        end_date=END,
        product_type="RSLC",
        path_number=PATH,
    )
    print(f"found {len(urls)} RSLC URLs on path {PATH}")
    for u in urls:
        print("  ", u.rsplit("/", 1)[-1])

    if len(urls) == 0:
        raise SystemExit("No RSLCs returned -- check AOI / date window.")

    # Step 2: download in parallel, skipping any already on disk.
    # max_workers kept modest because RSLCs are large (multi-GB).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\ndownloading to: {OUT_DIR}")
    paths = download_urls(urls, OUT_DIR, max_workers=2)

    print(f"\ndone -- {len(paths)} files present in {OUT_DIR.name}/")
    for p in paths:
        size_gb = p.stat().st_size / 1e9
        print(f"  {p.name}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
