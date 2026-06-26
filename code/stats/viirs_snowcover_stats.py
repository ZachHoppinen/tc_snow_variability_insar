"""viirs_snowcover_stats.py -- VIIRS fSCA vs the corrected gamma_20-gamma_80
(Appendix B, Figure~\\ref{fig:viirs}).

Source of truth for the Appendix B numbers. Builds, on the WY2023 81 m footprint
(the same display footprint as Figure~\\ref{fig:coherence_rows}): the per-pair
VIIRS fractional snow-covered area (fSCA) on each pair's end date and the
corrected gamma_20-gamma_80 residual. main() prints the per-pair Pearson/Spearman
correlation and mean fSCA quoted in the appendix; the figure script
(visualizations/plot_viirs_snowcover.py) imports build_maps() and only plots, so
the numbers and the figure stay in sync.

fSCA is derived from the VNP10A1F NDSI snow cover (0-100) via the Salomonson &
Appel (2004) relation fSCA = -0.01 + 1.45 * NDSI, clipped to [0, 1]. We use each
pair's end date because the decorrelation in an accumulation pair is driven by
the new snow, which is only fully on the ground at the end of the baseline.

Run download/fetch_viirs_snowcover.py first to populate raw/viirs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  registers .rio
from rasterio.enums import Resampling
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # code/ dir
sys.path.insert(0, str(HERE.parent / "visualizations"))
from tc_paths import DATA_ROOT  # noqa: E402
from plot_coherence_rows import load_absM, ad_diff  # noqa: E402  (shared loaders)

VIIRS_DIR = DATA_ROOT / "sub_pixel_variability/raw/viirs"
BAND = "L"
WY2023 = ("2023Mar02-03", "2023Mar16-17")             # largest ASO footprint
# (pair prefix, viirs tag, end date, row label) -- date must match
# viirs/fetch_viirs_snowcover.py
ROWS = [("P12", "p12", "2025-12-19",
         "P12  Dec 7-19, 2025\n(no accumulation)"),
        ("P23", "p23", "2025-12-31",
         "P23  Dec 19-31, 2025\n(large storm)")]


def _viirs_fsca(tag: str, date: str, template) -> np.ndarray:
    """VIIRS fSCA (0-1) on the template grid for a pair's end date. NDSI values
    outside [0, 100] (cloud/water/no-decision flags) are dropped, then NDSI is
    converted to fSCA via Salomonson & Appel (2004), clipped to [0, 1]."""
    v = (rioxarray.open_rasterio(VIIRS_DIR / f"viirs_{tag}_{date}.tif",
                                 masked=True).squeeze(drop=True))
    ndsi = v.rio.reproject_match(template, resampling=Resampling.nearest).values
    ndsi = np.where((ndsi >= 0) & (ndsi <= 100), ndsi, np.nan)
    return np.clip(-0.01 + 1.45 * (ndsi / 100.0), 0.0, 1.0)


def build_maps():
    """Return (template, mask, maps) where maps[prefix] = {'fsca', 'dgamma'} on
    the WY2023 footprint. `template` is the WY2023 |M| DataArray (display grid)."""
    template = load_absM(WY2023, BAND)
    mask = np.isfinite(template.values)               # WY2023 snow footprint
    maps = {}
    for prefix, tag, date, _ in ROWS:
        fsca = np.where(mask, _viirs_fsca(tag, date, template), np.nan)
        dgamma = np.where(mask, ad_diff(prefix, template), np.nan)
        maps[prefix] = {"fsca": fsca, "dgamma": dgamma}
    return template, mask, maps


def main() -> None:
    _, _, maps = build_maps()
    for prefix, _, _, _ in ROWS:
        fsca, dg = maps[prefix]["fsca"], maps[prefix]["dgamma"]
        m = np.isfinite(fsca) & np.isfinite(dg)
        s, d = fsca[m], dg[m]
        r, _ = pearsonr(s, d)
        rho, _ = spearmanr(s, d)
        print(f"{prefix}: n={m.sum():,}  Pearson r={r:+.3f}  Spearman rho={rho:+.3f}"
              f"  fSCA mean={s.mean():.2f}  dgamma median={np.median(d):+.3f}")
    print("\nNote: the ~4.4e5 pixels are strongly spatially autocorrelated, so "
          "per-pixel p-values are meaningless and are not reported.")


if __name__ == "__main__":
    main()
