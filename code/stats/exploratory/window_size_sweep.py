"""window_size_sweep.py -- how the sub-window coherence factor |M| varies with
the multilook WINDOW SIZE, computed on the NATIVE 3 m ASO dSWE field.

Unlike the main M pipeline (which block-averages 3 m -> 9 m before windowing to
beat down ASO depth noise), this script runs straight on the 3 m field. We
deliberately accept the extra per-pixel ASO noise so that we can resolve small
windows (9 m, 21 m, ...) and watch |M| fall as the footprint grows. The 9 m
pre-average would otherwise blur out exactly the small-window behavior we want
to see here.

|M| is computed exactly as elsewhere:
    M = < exp(i * kappa * dSWE) > * exp(-i * kappa * <dSWE>)
over non-overlapping square windows of side `b` 3 m cells. We sweep `b` to give
window sizes from 3 m (a single cell, where |M| = 1 by construction) up past the
80 m GUNW footprint, and report median |M| (with IQR) per window size, per pair.

CAVEAT: because there is no pre-average, the small-window |M| carries ASO depth
noise as well as true sub-window dSWE structure, so the small-window loss is an
over-estimate of the real sub-window loss. The curve still shows the SHAPE of
how fast |M| degrades with footprint, which is what informs a processing-window
choice. L-band only.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-slice windows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "ASO"))
from aso_M_calculation import kappa_of, BANDS, MIN_FRAC, BASE_RES_M  # noqa: E402

# repo-relative roots (code/stats/exploratory/ -> code/ is two levels up)
sys.path.insert(0, str(HERE.parents[1]))
from tc_paths import DATA_ROOT, FIG_ROOT  # noqa: E402

PROCESSED_DIR = DATA_ROOT / "sub_pixel_variability/processed/aso"
MASK_TIF = PROCESSED_DIR / "common_aso_mask.tif"   # common snow pixels, 4 pairs
FIG_DIR = FIG_ROOT / "stats"
PAIRS = [
    ("WY2023", "2023Mar02-03",    "2023Mar16-17"),
    ("WY2024", "2024Jan29",       "2024Feb27-28"),
    ("WY2025", "2025Feb08-09",    "2025Feb25"),
    ("WY2026", "2026Jan31-Feb01", "2026Feb27-28"),
]
KAPPA = kappa_of(BANDS["L"])          # L-band, the analysis band

# window side in 3 m cells -> window size in metres. Every 3 m step from 3 m
# (1 cell) up to 81 m (27 cells, ~ the NISAR GUNW footprint). 1 cell = 3 m is
# degenerate (|M| = 1, no within-window deviation) and anchors the curve.
SIDES = list(range(1, 28))                  # 1..27 cells
SIZES_M = [s * BASE_RES_M for s in SIDES]   # 3, 6, 9, ... 81 m


def load_dswe_3m(date_a: str, date_b: str) -> xr.DataArray:
    """Native 3 m dSWE field (metres) for one pair, no pre-average."""
    with xr.open_dataset(PROCESSED_DIR / f"{date_a}_{date_b}" / "aso_dswe_3m.nc") as ds:
        return ds["dswe"].load().astype("float32")


def build_or_load_common_mask() -> np.ndarray:
    """Common snow mask: pixels where ALL four pairs have valid (finite) dSWE.

    All four pairs share one 3 m UTM-11N grid, so this is a per-pixel AND of
    each pair's finite mask, no regridding. Built once and cached to
    `MASK_TIF` (uint8: 1 = common snow, 0 = not) for reuse, then returned as a
    boolean array. Comparing |M| over this single pixel set keeps the per-year
    curves on identical ground rather than each pair's own footprint."""
    if MASK_TIF.exists():
        with rioxarray.open_rasterio(MASK_TIF) as m:
            print(f"loaded common mask {MASK_TIF.name}")
            return m.squeeze(drop=True).values.astype(bool)

    common = None
    ref = None
    for wy, a, b in PAIRS:
        da = load_dswe_3m(a, b)
        if ref is None:
            ref = da                      # keep one DataArray for coords/CRS
        v = np.isfinite(da.values)
        common = v if common is None else (common & v)
        print(f"  {wy}: {v.sum():,} valid -> running common {common.sum():,}")

    # write the cached GeoTIFF on the shared grid (EPSG:32611, UTM 11N)
    out = xr.DataArray(common.astype("uint8"), dims=ref.dims, coords=ref.coords)
    out.rio.write_crs("EPSG:32611", inplace=True)
    out.rio.to_raster(MASK_TIF, compress="lzw")
    print(f"wrote {MASK_TIF}  ({common.sum():,} common snow pixels)")
    return common


def M_for_side(d: np.ndarray, side: int):
    """|M| and arg(M) over non-overlapping `side` x `side` windows of `d`.

    Keeps windows with at least MIN_FRAC of their cells valid (snow-covered).
    Computed with cos/sin sums in float32 to keep memory bounded on the full
    3 m array rather than materialising a complex field. Returns
    (|M|, arg(M) in radians) for the valid windows."""
    ny, nx = d.shape
    nby, nbx = ny // side, nx // side
    # trim to a whole number of windows and fold into (nby, side, nbx, side)
    w = d[:nby * side, :nbx * side].reshape(nby, side, nbx, side)
    valid = np.isfinite(w)
    cnt = valid.sum(axis=(1, 3))                       # valid cells per window
    d0 = np.where(valid, w, 0.0).astype("float32")     # NaNs -> 0 (masked by valid)

    theta = (KAPPA * d0).astype("float32")             # per-cell phase
    cos_sum = (np.cos(theta) * valid).sum(axis=(1, 3))
    sin_sum = (np.sin(theta) * valid).sum(axis=(1, 3))
    n = np.maximum(cnt, 1)
    zbar = (cos_sum + 1j * sin_sum) / n                 # window-mean phasor
    mean_d = d0.sum(axis=(1, 3)) / n                    # window-mean dSWE (m)
    M = zbar * np.exp(-1j * KAPPA * mean_d)             # rotate onto the true mean

    ok = cnt >= int(np.ceil(MIN_FRAC * side * side))
    return np.abs(M)[ok], np.angle(M)[ok]


def main() -> None:
    # med[wy] = median |M| per window size;  am_all[wy][si] = raw |M| array
    # (kept per size so the pooled median can be formed exactly across pairs)
    med = {wy: [] for wy, *_ in PAIRS}
    am_all = {wy: [] for wy, *_ in PAIRS}

    common = build_or_load_common_mask()    # bool, shared 3 m grid

    # load each pair's 3 m field ONCE, restrict to the common snow pixels, then
    # sweep every window size on it so all pairs use identical ground
    for wy, a, b in PAIRS:
        print(f"\n--- {wy}: loading 3 m dSWE ---")
        d = load_dswe_3m(a, b).values
        d[~common] = np.nan                 # keep only the common snow pixels
        for side in SIDES:
            am = M_for_side(d, side)[0]
            med[wy].append(np.median(am))
            am_all[wy].append(am)
            print(f"  {side * BASE_RES_M:3d} m ({side:2d}x{side:<2d}): "
                  f"median |M|={np.median(am):.3f}  n={am.size:,}")
        del d                                   # free the 2.2 GB field

    nsz = len(SIDES)
    # pooled median |M| across all four pairs at each window size (exact)
    pooled_med = [float(np.median(np.concatenate([am_all[wy][si]
                  for wy, *_ in PAIRS]))) for si in range(nsz)]

    print("\n=== median |M| vs window size (L-band) ===")
    print("  size(m)   pooled|M|")
    for si, size_m in enumerate(SIZES_M):
        print(f"  {size_m:5d}   {pooled_med[si]:9.3f}")

    # plot: median |M| vs window size, one line per pair + pooled
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for wy, *_ in PAIRS:
        ax.plot(SIZES_M, med[wy], "o-", label=wy)
    ax.plot(SIZES_M, pooled_med, "k--", lw=2, label="pooled")
    ax.axvline(80, color="gray", ls=":", lw=1)
    ax.text(80, 1.0, " 80 m GUNW", va="top", ha="right",
            fontsize=8, color="gray")
    ax.set_xticks([3, 9, 15, 21, 27, 39, 51, 63, 75, 81])
    ax.set_xlabel("multilook window size (m)")
    ax.set_ylabel("median |M|  (L-band)")
    ax.set_title("Sub-window coherence factor vs window size (native 3 m dSWE)\n"
                 "small windows carry ASO depth noise; 3 m = single cell (|M|=1)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "window_size_sweep_L.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
