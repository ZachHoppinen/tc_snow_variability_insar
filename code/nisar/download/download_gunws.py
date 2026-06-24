"""Download the 6 NISAR GUNWs on path 042 over the Tuolumne ASO lidar AOI.

Pulls from ASF via nisar_pytools and writes to
``nisar_swe/data/nisar/gunw/tuolome/``. Existing files are skipped.
Requires Earthdata Login credentials in ``~/.netrc``.
"""

from pathlib import Path

from nisar_pytools import download_urls, find_nisar

# Tuolumne ASO lidar bounds (EPSG:4326), from
# ASO_Tuolumne_2025Feb08-09_snowdepth_50m.tif.
AOI = [-120.288, 37.714, -119.191, 38.241]   # [W, S, E, N]

START = "2025-01-01"
END = "2026-05-13"
PATH = 42                                      # descending track 042
OUT_DIR = Path("/Users/zmhoppinen/Documents/nisar_swe/data/nisar/gunw/tuolome")


def main() -> None:
    # Step 1: query ASF for matching GUNW URLs.
    urls = find_nisar(
        AOI,
        start_date=START,
        end_date=END,
        product_type="GUNW",
        path_number=PATH,
    )
    print(f"found {len(urls)} GUNW URLs on path {PATH}")
    for u in urls:
        print("  ", u.rsplit("/", 1)[-1])

    # Step 2: download in parallel, skipping any already on disk.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\ndownloading to: {OUT_DIR}")
    paths = download_urls(urls, OUT_DIR, max_workers=4)

    print(f"\ndone -- {len(paths)} files present in {OUT_DIR.name}/")
    for p in paths:
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
