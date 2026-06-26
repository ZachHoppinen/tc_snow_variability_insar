"""viirs_snowcover_stats.py -- VIIRS snow cover vs the corrected gamma_20-gamma_80
(Appendix B, Figure~\\ref{fig:viirs}).

Source of truth for the Appendix B numbers. Builds, on the WY2023 81 m footprint
(the same display footprint as Figure~\\ref{fig:coherence_rows}): the per-pair
VIIRS NDSI snow cover (mean over each pair's three download dates) and the
corrected gamma_20-gamma_80 residual. main() prints the per-pair Pearson/Spearman
correlation and mean snow cover quoted in the appendix; the figure script
(visualizations/plot_viirs_snowcover.py) imports build_maps() and only plots, so
the numbers and the figure stay in sync.

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
# (pair prefix, viirs tag, window dates, row label) -- dates must match
# viirs/fetch_viirs_snowcover.py
ROWS = [("P12", "p12", ["2025-12-07", "2025-12-13", "2025-12-19"],
         "P12  Dec 7-19, 2025\n(no accumulation)"),
        ("P23", "p23", ["2025-12-19", "2025-12-25", "2025-12-31"],
         "P23  Dec 19-31, 2025\n(large storm)")]


def _viirs_mean(tag: str, dates: list[str], template) -> np.ndarray:
    """Mean VIIRS NDSI snow cover over a pair's dates, on the template grid.
    NDSI values outside [0, 100] (cloud/water/no-decision flags) are dropped."""
    stack = []
    for d in dates:
        v = (rioxarray.open_rasterio(VIIRS_DIR / f"viirs_{tag}_{d}.tif",
                                     masked=True).squeeze(drop=True))
        on = v.rio.reproject_match(template, resampling=Resampling.nearest).values
        stack.append(np.where((on >= 0) & (on <= 100), on, np.nan))
    return np.nanmean(np.stack(stack), axis=0)


def build_maps():
    """Return (template, mask, maps) where maps[prefix] = {'snow', 'dgamma'} on
    the WY2023 footprint. `template` is the WY2023 |M| DataArray (display grid)."""
    template = load_absM(WY2023, BAND)
    mask = np.isfinite(template.values)               # WY2023 snow footprint
    maps = {}
    for prefix, tag, dates, _ in ROWS:
        snow = np.where(mask, _viirs_mean(tag, dates, template), np.nan)
        dgamma = np.where(mask, ad_diff(prefix, template), np.nan)
        maps[prefix] = {"snow": snow, "dgamma": dgamma}
    return template, mask, maps


def main() -> None:
    _, _, maps = build_maps()
    for prefix, _, _, _ in ROWS:
        snow, dg = maps[prefix]["snow"], maps[prefix]["dgamma"]
        m = np.isfinite(snow) & np.isfinite(dg)
        s, d = snow[m], dg[m]
        r, _ = pearsonr(s, d)
        rho, _ = spearmanr(s, d)
        print(f"{prefix}: n={m.sum():,}  Pearson r={r:+.3f}  Spearman rho={rho:+.3f}"
              f"  snow mean={s.mean():.0f}%  dgamma median={np.median(d):+.3f}")
    print("\nNote: the ~4.4e5 pixels are strongly spatially autocorrelated, so "
          "per-pixel p-values are meaningless and are not reported.")


if __name__ == "__main__":
    main()
