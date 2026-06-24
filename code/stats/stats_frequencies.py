"""stats_frequencies.py -- per-band (C/L/P) |M| and arg(M) statistics, for
writing up the wavelength comparison (Fig. bands_M_WY2025).

Reads the per-pair, per-band M files written by aso_M_calculation.py and reports,
for each band, the |M| and arg(M) distributions over the 81 m windows. Done for
the figure's WY2025 pair and pooled over all four pairs. Low |M| already implies
a randomized (undefined) phase bias, so we report |M| (median, IQR, fraction
below 0.1) and arg(M) (median, IQR) -- no separate phase-randomness metric.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-slice windows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ASO"))
from aso_M_calculation import kappa_of, PREAVG, BLOCK, MIN_FRAC, BANDS as WL  # noqa: E402

PROCESSED_DIR = Path("/Users/zmhoppinen/Documents/nisar_swe/data/"
                     "sub_pixel_variability/processed/aso")
MIN_CELLS = int(np.ceil(MIN_FRAC * BLOCK * BLOCK))
PAIRS = [
    ("WY2023", "2023Mar02-03",    "2023Mar16-17"),
    ("WY2024", "2024Jan29",       "2024Feb27-28"),
    ("WY2025", "2025Feb08-09",    "2025Feb25"),
    ("WY2026", "2026Jan31-Feb01", "2026Feb27-28"),
]
FOCUS = "WY2025"                      # the pair shown in the figure
BANDS = ["C", "L", "P"]
KAPPA = {"C": 224.1, "L": 52.7, "P": 15.5}   # rad/m at 40 deg (for the print)


def load_band(date_a: str, date_b: str, band: str):
    """(|M|, arg(M) deg) finite arrays for one pair and band."""
    pdir = PROCESSED_DIR / f"{date_a}_{date_b}"
    with xr.open_dataset(pdir / f"aso_M_81m_{band}.nc") as ds:
        M = ds["M_real"].values + 1j * ds["M_imag"].values
    fin = np.isfinite(M)
    return np.abs(M[fin]), np.degrees(np.angle(M[fin]))


def report(tag: str, absM: np.ndarray, argM: np.ndarray) -> None:
    p25, p50, p75 = np.percentile(absM, [25, 50, 75])
    a25, a50, a75 = np.percentile(argM, [25, 50, 75])
    frac_lo = np.mean(absM < 0.1) * 100.0                 # ~decorrelated windows
    print(f"  {tag:10s} |M| med={p50:.3f} IQR=[{p25:.3f},{p75:.3f}]  "
          f"|M|<0.1: {frac_lo:4.1f}%  | "
          f"arg med={a50:+5.1f} IQR=[{a25:+6.1f},{a75:+6.1f}]")


def window_field_stats(date_a: str, date_b: str):
    """Per-81 m-window within-window dSWE skewness and per-band arg(M), from the
    3 m dSWE field (3x3 -> 9 m, then 9x9 windows). Returns (skew, {band: argM})
    over valid windows, all aligned to the same windows."""
    pdir = PROCESSED_DIR / f"{date_a}_{date_b}"
    with xr.open_dataset(pdir / "aso_dswe_3m.nc") as ds:
        d3 = ds["dswe"].load()
    d9 = d3.coarsen(x=PREAVG, y=PREAVG, boundary="trim").mean(skipna=True).values

    ny, nx = d9.shape
    nby, nbx = ny // BLOCK, nx // BLOCK
    b = d9[:nby * BLOCK, :nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)  # m
    valid = np.isfinite(b)
    cnt = valid.sum(axis=(1, 3))

    mean = np.nanmean(b, axis=(1, 3))
    dev = b - mean[:, None, :, None]
    std = np.sqrt(np.nanmean(dev ** 2, axis=(1, 3)))
    skew = np.nanmean(dev ** 3, axis=(1, 3)) / std ** 3        # standardized 3rd moment

    args = {}
    for band in BANDS:
        k = kappa_of(WL[band])
        z = np.where(valid, np.exp(1j * k * np.nan_to_num(b)), 0.0)
        zbar = z.sum(axis=(1, 3)) / np.maximum(cnt, 1)
        M = zbar * np.exp(-1j * k * np.nan_to_num(mean))
        args[band] = np.degrees(np.angle(M))

    ok = (cnt >= MIN_CELLS) & np.isfinite(skew)
    return skew[ok], {band: args[band][ok] for band in BANDS}


def main() -> None:
    fa, fb = next((a, b) for wy, a, b in PAIRS if wy == FOCUS)

    print("### per-band |M| and arg(M) ###")
    for band in BANDS:
        print(f"\n=== {band}-band (kappa = {KAPPA[band]} rad/m) ===")
        am, ag = load_band(fa, fb, band)
        report(FOCUS, am, ag)
        pooled_a = np.concatenate([load_band(a, b, band)[0] for _, a, b in PAIRS])
        pooled_g = np.concatenate([load_band(a, b, band)[1] for _, a, b in PAIRS])
        report("POOLED", pooled_a, pooled_g)

    # within-window dSWE skewness, and whether it drives a negative bias
    print("\n### within-window dSWE skewness -> phase bias ###")
    sk = {}                       # tag -> (skew array, {band: argM})
    sk[FOCUS] = window_field_stats(fa, fb)
    allsk = [window_field_stats(a, b) for _, a, b in PAIRS]
    sk["POOLED"] = (np.concatenate([s for s, _ in allsk]),
                    {band: np.concatenate([a[band] for _, a in allsk])
                     for band in BANDS})
    for tag, (skew, argd) in sk.items():
        pos = np.mean(skew > 0) * 100.0
        print(f"\n  [{tag}]  within-window dSWE skew: "
              f"median={np.median(skew):+.2f}, mean={skew.mean():+.2f}, "
              f"{pos:.0f}% positive  (n={skew.size:,})")
        for band in BANDS:
            a = argd[band]
            r = np.corrcoef(skew, a)[0, 1]
            print(f"      {band}-band arg(M): median={np.median(a):+5.1f} deg, "
                  f"mean={a.mean():+5.1f}  | corr(skew, arg) = {r:+.3f}")


if __name__ == "__main__":
    main()
