"""consistency_stats.py -- spatial consistency of |M| and arg(M) across the four
ASO pairs, to support "the four pairs show consistent spatial patterns."

All pairs share the 81 m grid, so we correlate the maps pixel-for-pixel:
  - individual: pairwise r^2 between each pair of pairs (6 combinations)
  - pooled: r^2 of each pair against the per-pixel mean over all four pairs
A high mean pairwise r^2, and higher still against the pooled mean, indicates a
shared, terrain-locked spatial pattern that recurs year to year.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo code/ dir
from tc_paths import DATA_ROOT  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

SMOOTH_SIGMA = 3            # px (~240 m at 81 m) for the smoothed-arg test

PROCESSED_DIR = DATA_ROOT / "sub_pixel_variability/processed/aso"
PAIRS = [("WY2023", "2023Mar02-03_2023Mar16-17"),
         ("WY2024", "2024Jan29_2024Feb27-28"),
         ("WY2025", "2025Feb08-09_2025Feb25"),
         ("WY2026", "2026Jan31-Feb01_2026Feb27-28")]
BAND = "L"


def load_maps(folder: str):
    """(|M| map, arg(M)-deg map) as 2-D arrays on the shared 81 m grid."""
    with xr.open_dataset(PROCESSED_DIR / folder / f"aso_M_81m_{BAND}.nc") as ds:
        M = ds["M_real"].values + 1j * ds["M_imag"].values
    return np.abs(M), np.degrees(np.angle(M))


def r2(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1] ** 2)


def nan_smooth(a: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """NaN-aware Gaussian smooth (suppresses per-pixel noise, keeps the mask)."""
    fin = np.isfinite(a)
    num = gaussian_filter(np.where(fin, a, 0.0), sigma)
    den = gaussian_filter(fin.astype(float), sigma)
    out = np.where(den > 0.1, num / np.maximum(den, 1e-9), np.nan)
    out[~fin] = np.nan
    return out


def consistency(field_name: str, maps: dict) -> None:
    wys = [wy for wy, _ in PAIRS]
    print(f"\n=== {field_name}: spatial consistency across pairs ===")

    print("  individual pairwise r^2:")
    pair_r2 = []
    for i, j in combinations(wys, 2):
        v = r2(maps[i], maps[j])
        pair_r2.append(v)
        print(f"    {i} vs {j}: {v:.3f}")
    print(f"  mean pairwise r^2 = {np.mean(pair_r2):.3f}")

    # pooled = per-pixel mean over the four maps
    pooled = np.nanmean(np.stack([maps[wy] for wy in wys]), axis=0)
    print("  each pair vs pooled-mean pattern r^2:")
    for wy in wys:
        print(f"    {wy} vs pooled: {r2(maps[wy], pooled):.3f}")


def main() -> None:
    maps = {wy: load_maps(folder) for wy, folder in PAIRS}
    absM = {wy: maps[wy][0] for wy, _ in PAIRS}
    argM = {wy: maps[wy][1] for wy, _ in PAIRS}
    absarg = {wy: np.abs(maps[wy][1]) for wy, _ in PAIRS}
    smarg = {wy: nan_smooth(maps[wy][1]) for wy, _ in PAIRS}

    consistency("|M|", absM)
    consistency("arg(M) signed (per-pixel)", argM)
    consistency("|arg(M)|  (bias magnitude)", absarg)
    consistency(f"arg(M) smoothed (sigma={SMOOTH_SIGMA}px, signed)", smarg)


if __name__ == "__main__":
    main()
