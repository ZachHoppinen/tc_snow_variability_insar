"""Sub-window coherence loss |M| and phase bias arg(M) vs multilook window size.

Idea: the sub-window penalty is usually shown at one footprint (81 m). Here we
sweep the multilook window from 9 m to the 80 m GUNW size and watch both effects
develop, from the ASO dSWE (no model). For a square window of side W the per-cell
SWE phase is phi = kappa*dSWE and

    M = < exp(i*kappa*dSWE) > * exp(-i*kappa*<dSWE>),

so |M| in [0, 1] is the coherence retained and arg(M) the residual phase bias
from the within-window dSWE distribution. As W grows the window swallows more
sub-window variability, so |M| falls and arg(M) carries the skew-driven bias.

The sweep runs on the native 3 m dSWE (no 9 m pre-average) so windows as small
as 9 m are resolved. This folds per-pixel ASO depth noise into |M|, making the
plotted |M| a LOWER BOUND on the true coherence; that noise is quantified in the
manuscript limitations. The window construction (non-overlapping squares, half
the cells snow-covered) matches the main M analysis.

All four ASO pairs are pooled over one common snow mask (identical pixels every
pair) and summarised as median and IQR. arg(M) is shown for usable windows only
(|M| > USABLE_M), mirroring operational coherence masking; below that threshold
arg(M) is near-uniform over (-pi, pi] and carries no recoverable signal.

Two stacked panels: top |M|, bottom arg(M) in L-band mm SWE. L-band (NISAR).
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

# canonical SWE-phase constant and window parameters
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ASO"))
from aso_M_calculation import kappa_of, BANDS, MIN_FRAC, BASE_RES_M  # noqa: E402
from plotting_constants import (  # noqa: E402
    apply_style, DPI, PROCESSED_DIR, FIG_DIR, PAIRS, COLOR_M, COLOR_ARG)

MASK_TIF = PROCESSED_DIR / "common_aso_mask.tif"   # common snow pixels, 4 pairs
BAND = "L"                                   # NISAR L-band
KAPPA = kappa_of(BANDS[BAND])                # rad per m SWE
SIDES = list(range(3, 28))                   # 3..27 cells of 3 m -> 9..81 m
SIZES_M = np.array([s * BASE_RES_M for s in SIDES])
USABLE_M = 0.3                               # coherence cut for the bias panel
M_EDGES = np.linspace(0.0, 1.0, 1001)        # |M| histogram bins
A_EDGES = np.linspace(-np.pi, np.pi, 1801)   # arg(M) histogram bins (rad)
GUNW_LINES = [20, 80]                        # NISAR GUNW postings to annotate


def load_dswe_3m(date_a: str, date_b: str) -> xr.DataArray:
    """Native 3 m dSWE field (metres) for one pair."""
    with xr.open_dataset(PROCESSED_DIR / f"{date_a}_{date_b}" / "aso_dswe_3m.nc") as ds:
        return ds["dswe"].load().astype("float32")


def build_or_load_common_mask() -> np.ndarray:
    """Common snow mask: pixels valid (finite dSWE) in ALL four pairs. The four
    pairs share one 3 m UTM-11N grid, so this is a per-pixel AND, no regridding.
    Cached to MASK_TIF (uint8) and returned as bool."""
    if MASK_TIF.exists():
        with rioxarray.open_rasterio(MASK_TIF) as m:
            return m.squeeze(drop=True).values.astype(bool)
    common, ref = None, None
    for _, a, b in PAIRS:
        da = load_dswe_3m(a, b)
        ref = da if ref is None else ref
        v = np.isfinite(da.values)
        common = v if common is None else (common & v)
    out = xr.DataArray(common.astype("uint8"), dims=ref.dims, coords=ref.coords)
    out.rio.write_crs("EPSG:32611", inplace=True)
    out.rio.to_raster(MASK_TIF, compress="lzw")
    return common


def M_for_side(d: np.ndarray, side: int):
    """(|M|, arg(M) [rad]) over non-overlapping side x side windows of `d`.
    Cos/sin sums in float32 keep memory bounded on the full 3 m array."""
    ny, nx = d.shape
    nby, nbx = ny // side, nx // side
    w = d[:nby * side, :nbx * side].reshape(nby, side, nbx, side)
    valid = np.isfinite(w)
    cnt = valid.sum(axis=(1, 3))
    d0 = np.where(valid, w, 0.0).astype("float32")
    theta = (KAPPA * d0).astype("float32")
    cos_sum = (np.cos(theta) * valid).sum(axis=(1, 3))
    sin_sum = (np.sin(theta) * valid).sum(axis=(1, 3))
    n = np.maximum(cnt, 1)
    zbar = (cos_sum + 1j * sin_sum) / n
    mean_d = d0.sum(axis=(1, 3)) / n
    M = zbar * np.exp(-1j * KAPPA * mean_d)              # rotate onto the true mean
    ok = cnt >= int(np.ceil(MIN_FRAC * side * side))     # >= half the window valid
    return np.abs(M)[ok], np.angle(M)[ok]


def pctiles_from_hist(counts, edges, qs):
    """Percentiles (qs in %) from a histogram via CDF interpolation."""
    c = np.cumsum(counts)
    if c[-1] == 0:
        return [np.nan] * len(qs)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return list(np.interp([q / 100.0 for q in qs], c / c[-1], centers))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    common = build_or_load_common_mask()

    # pool the four pairs by accumulating per-size histograms (so percentiles
    # come from the full window population without holding every window)
    nsz = len(SIDES)
    Mhist = [np.zeros(len(M_EDGES) - 1) for _ in range(nsz)]
    Ahist = [np.zeros(len(A_EDGES) - 1) for _ in range(nsz)]
    for wy, a, b in PAIRS:
        print(f"--- {wy} ---")
        d = load_dswe_3m(a, b).values
        d[~common] = np.nan
        for si, s in enumerate(SIDES):
            am, ag = M_for_side(d, s)
            Mhist[si] += np.histogram(am, bins=M_EDGES)[0]
            Ahist[si] += np.histogram(ag[am > USABLE_M], bins=A_EDGES)[0]
        del d

    Mp = np.array([pctiles_from_hist(Mhist[si], M_EDGES, [25, 50, 75])
                   for si in range(nsz)])
    Ap = np.array([pctiles_from_hist(Ahist[si], A_EDGES, [25, 50, 75])
                   for si in range(nsz)]) / KAPPA * 1000.0     # rad -> mm SWE

    fig, (axM, axA) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True,
                                   constrained_layout=True)

    axM.fill_between(SIZES_M, Mp[:, 0], Mp[:, 2], color=COLOR_M, alpha=0.2,
                     label="IQR")
    axM.plot(SIZES_M, Mp[:, 1], "o-", color=COLOR_M, lw=2, ms=5, label="median")
    axM.set_ylabel("|M|  (coherence factor)")
    axM.set_ylim(0, 1.0)
    axM.legend(loc="upper right", frameon=False)

    axA.fill_between(SIZES_M, Ap[:, 0], Ap[:, 2], color=COLOR_ARG, alpha=0.2)
    axA.plot(SIZES_M, Ap[:, 1], "s-", color=COLOR_ARG, lw=2, ms=5)
    axA.axhline(0, color="0.5", lw=0.8)
    axA.set_ylabel(r"arg(M)  (L-band mm SWE)")
    axA.set_xlabel("multilook window size (m)")

    # annotate the NISAR GUNW postings
    for ax in (axM, axA):
        for gw in GUNW_LINES:
            ax.axvline(gw, color="0.6", ls=":", lw=1)
    for gw in GUNW_LINES:
        axM.text(gw, 0.02, f" {gw} m", rotation=90, va="bottom", ha="right",
                 fontsize=9, color="0.4")

    out = FIG_DIR / f"window_M_arg_vs_size_{BAND}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
