"""aso_noise.py -- how much of the |M| drop could be ASO depth noise, not real
sub-pixel dSWE structure?

The |M| pipeline runs on the 9 m (3x3 block-averaged) dSWE field. Any per-pixel
ASO depth error that survives that average adds its own within-window phase
spread and deflates |M| exactly the way real sub-pixel dSWE variability does.
Because kappa is ~4.3x larger at C-band than L-band, the same residual depth
error decorrelates much harder at the shorter wavelength, so the confound is
worst in the band where we report the strongest collapse.

We estimate the NOISE-ONLY |M| floor empirically by measuring |M| in SMOOTH,
NEAR-ZERO-CHANGE windows (window-mean |dSWE| small AND the window-mean field
locally flat), and -- the key step -- DETRENDING each window before measuring
the residual. We fit and subtract a linear plane over the 9 m cells in each
window, so the real smooth dSWE gradient (low-frequency, the red/fractal snow
signal) is removed and what remains is the high-frequency, pixel-uncorrelated
ASO laser-shot noise. Computing |M| on that detrended residual gives the noise
floor: what |M| would read if a window held only ASO noise and no real signal.

Why detrend HERE but not in the main |M| analysis: the main analysis must keep
the real within-window gradient, because a genuine smooth gradient across the
multilook window really does decorrelate the InSAR observation. The noise
sub-analysis wants the opposite -- to strip the real signal and leave only the
laser-shot noise -- so we remove the trend. (This does not reintroduce the
block-detrend variogram-range artifact: that affected correlation-LENGTH
estimation; here we only use the residual VARIANCE as a noise proxy.)

We report both the mean-removed spread (real gradient still in: an UPPER bound
on noise) and the detrended spread (gradient removed: the ASO shot-noise floor).

Output: per band, the observed median |M| (all valid windows, signal+noise) next
to the noise-only median |M| (calm windows), per pair and pooled. The pooled
noise-only numbers are what fills the depth-noise limitation paragraph.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ASO"))
sys.path.insert(0, str(HERE.parent))  # repo code/ dir
from aso_M_calculation import (  # noqa: E402
    BANDS, BLOCK, MIN_FRAC, PAIRS, PROCESSED_DIR,
    kappa_of, load_dswe, preaverage, window_M,
)
from tc_paths import FIG_ROOT  # noqa: E402

FIG_DIR = FIG_ROOT / "stats"

MIN_CELLS = int(np.ceil(MIN_FRAC * BLOCK * BLOCK))   # >= 41 of 81 cells valid

# --- "smooth, near-zero-change" selection (the noise-only region) -----------
# Tunable; sanity-check the count and the highlighted region in the figure.
LOW_CHANGE_M = 0.03    # |window-mean dSWE| < 3 cm  -> little real change
SMOOTH_M = 0.02        # local std of the window-mean field < 2 cm -> flat
FIG_PAIR = "WY2025"    # which pair to draw the diagnostic for


def coarse_window_fields(d9: np.ndarray):
    """Per-window mean dSWE, within-window sigma, and valid-cell count.

    Reshapes the 9 m field into the SAME non-overlapping BLOCKxBLOCK (81 m)
    windows window_M() uses, so the outputs align cell-for-cell with the |M|
    fields. All band-independent (no kappa here).
    """
    ny, nx = d9.shape
    nby, nbx = ny // BLOCK, nx // BLOCK
    d = d9[:nby * BLOCK, :nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    valid = np.isfinite(d)
    cnt = valid.sum(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        mean_d = np.nanmean(d, axis=(1, 3))                       # m
        dev = d - mean_d[:, None, :, None]
        sigma = np.sqrt(np.nanmean(dev ** 2, axis=(1, 3)))       # m
    return mean_d, sigma, cnt


def local_std(a: np.ndarray) -> np.ndarray:
    """NaN-aware std of each cell's 3x3 window-neighborhood.

    Measures how much the window-mean dSWE varies over a ~3-window (243 m)
    neighborhood: small = locally flat (smooth), large = a real gradient runs
    through here. Pure numpy: pad with NaN, stack the nine shifts, nanstd.
    """
    ap = np.pad(a, 1, constant_values=np.nan)
    nby, nbx = a.shape
    stack = np.stack([ap[i:i + nby, j:j + nbx]
                      for i in range(3) for j in range(3)])
    return np.nanstd(stack, axis=0)


def calm_mask(mean_d: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Windows that are valid, low-change, and locally flat (the noise region)."""
    valid = cnt >= MIN_CELLS
    low_change = np.abs(mean_d) < LOW_CHANGE_M
    smooth = local_std(mean_d) < SMOOTH_M
    return valid & low_change & smooth


# design matrix for a linear plane (1, x, y) over the BLOCKxBLOCK cell grid
_YY, _XX = np.mgrid[0:BLOCK, 0:BLOCK]
_PLANE = np.column_stack([np.ones(BLOCK * BLOCK),
                          _XX.ravel().astype(float),
                          _YY.ravel().astype(float)])


def detrended_noise(d9: np.ndarray, calm: np.ndarray, kappas: dict):
    """For each calm window, remove a fitted linear plane and measure the
    residual (ASO shot noise). Returns the per-window detrended sigma (m) and a
    dict of per-band noise-only |M| computed on the detrended residual.

    Linear detrend strips the real smooth dSWE gradient (low-frequency snow
    signal); the residual is the high-frequency, pixel-uncorrelated laser
    noise. |M|_noise = |<exp(i kappa * residual)>| over the window cells.
    """
    ny, nx = d9.shape
    nby, nbx = ny // BLOCK, nx // BLOCK
    d = d9[:nby * BLOCK, :nbx * BLOCK]

    sig, absM = [], {b: [] for b in kappas}
    for by, bx in zip(*np.where(calm)):
        blk = d[by * BLOCK:(by + 1) * BLOCK,
                bx * BLOCK:(bx + 1) * BLOCK].ravel()
        m = np.isfinite(blk)
        if m.sum() < MIN_CELLS:
            continue
        coef, *_ = np.linalg.lstsq(_PLANE[m], blk[m], rcond=None)
        resid = blk[m] - _PLANE[m] @ coef        # mean + plane removed
        sig.append(np.sqrt(np.mean(resid ** 2)))
        for b, k in kappas.items():
            absM[b].append(np.abs(np.mean(np.exp(1j * k * resid))))
    return np.array(sig), {b: np.array(v) for b, v in absM.items()}


def pair_arrays(date_a: str, date_b: str):
    """For one pair: calm/valid masks, mean_d, the mean-removed within-window
    sigma, per-band full-field |M|, and the detrended (shot-noise) sigma and
    per-band noise-only |M| over the calm windows."""
    d9 = preaverage(load_dswe(date_a, date_b))          # 3 m -> 9 m (same pipeline)
    mean_d, sigma, cnt = coarse_window_fields(d9.values)
    calm = calm_mask(mean_d, cnt)
    valid = cnt >= MIN_CELLS

    absM = {}
    for band, lam in BANDS.items():
        M = window_M(d9, kappa_of(lam))                 # full-field complex M (mean-removed)
        absM[band] = np.abs(M.values)

    kappas = {b: kappa_of(lam) for b, lam in BANDS.items()}
    dt_sig, dt_absM = detrended_noise(d9.values, calm, kappas)   # shot-noise floor
    return calm, valid, mean_d, sigma, absM, dt_sig, dt_absM


def summarize_pair(name, calm, valid, sigma, absM, dt_sig, dt_absM):
    """Per band: observed median |M| (all windows) next to the mean-removed and
    the detrended (shot-noise) noise-only median |M| over calm windows."""
    n_calm = int(calm.sum())
    sig_mean = float(np.nanmean(sigma[calm])) * 100.0   # mean-removed (gradient in)
    sig_dt = float(np.mean(dt_sig)) * 100.0             # detrended (shot noise)
    print(f"\n{name}:  calm windows n={n_calm:,}   "
          f"within-window sigma: mean-removed {sig_mean:.2f} cm, "
          f"detrended {sig_dt:.2f} cm (9 m cells)")
    print(f"  {'band':>5s} {'kappa':>7s} {'obs med |M|':>12s} "
          f"{'noise |M| meanrm':>17s} {'noise |M| detrend':>18s}")
    for band, lam in BANDS.items():
        k = kappa_of(lam)
        obs = np.nanmedian(absM[band][valid])
        n_mean = np.nanmedian(absM[band][calm])
        n_dt = np.median(dt_absM[band])
        print(f"  {band:>5s} {k:7.1f} {obs:12.3f} {n_mean:17.3f} {n_dt:18.3f}")
    return n_calm


def make_figure(name, calm, valid, mean_d, absM, dt_absM):
    """Diagnostic: where the calm windows are, and the detrended shot-noise |M|
    histograms at C and L vs the observed medians."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5),
                                   constrained_layout=True)

    # (left) window-mean dSWE with the calm (noise) windows outlined
    md = np.where(valid, mean_d * 100.0, np.nan)
    im = ax0.imshow(md, cmap="RdBu", vmin=-30, vmax=30, origin="upper")
    fig.colorbar(im, ax=ax0, shrink=0.8, label="window-mean dSWE (cm)")
    ax0.imshow(np.where(calm, 1.0, np.nan), cmap="Greens", alpha=0.6,
               vmin=0, vmax=1, origin="upper")
    ax0.set_title(f"{name}: window-mean dSWE\n(green = calm/noise windows)")
    ax0.set_xlabel("window col"); ax0.set_ylabel("window row")

    # (right) detrended shot-noise |M| at C and L vs observed medians
    bins = np.linspace(0, 1, 41)
    for band, color in (("C", "tab:blue"), ("L", "tab:green")):
        ax1.hist(dt_absM[band], bins=bins, density=True, histtype="step",
                 color=color, lw=2, label=f"{band} detrended noise |M|")
        ax1.axvline(np.nanmedian(absM[band][valid]), color=color, ls="--",
                    lw=1.5, label=f"{band} observed median (all)")
    ax1.set_xlabel("|M|"); ax1.set_ylabel("density")
    ax1.set_title("detrended shot-noise |M| vs observed median")
    ax1.legend(fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"aso_noise_floor_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


def main() -> None:
    print(f"noise region: |window-mean dSWE| < {LOW_CHANGE_M*100:.0f} cm AND "
          f"local mean-field std < {SMOOTH_M*100:.0f} cm")
    print("(pessimistic noise bound: calm windows may hold some real signal, "
          "so the true noise drop is no worse than shown)")

    pooled = {"valid": {b: [] for b in BANDS},
              "noise_meanrm": {b: [] for b in BANDS},
              "noise_dt": {b: [] for b in BANDS},
              "sig_meanrm": [], "sig_dt": []}

    for wy, a, b in PAIRS:
        calm, valid, mean_d, sigma, absM, dt_sig, dt_absM = pair_arrays(a, b)
        summarize_pair(wy, calm, valid, sigma, absM, dt_sig, dt_absM)
        for band in BANDS:
            pooled["valid"][band].append(absM[band][valid])
            pooled["noise_meanrm"][band].append(absM[band][calm])
            pooled["noise_dt"][band].append(dt_absM[band])
        pooled["sig_meanrm"].append(sigma[calm])
        pooled["sig_dt"].append(dt_sig)
        if wy == FIG_PAIR:
            make_figure(wy, calm, valid, mean_d, absM, dt_absM)

    # pooled over all four pairs -- the numbers for the writeup
    sig_mean = float(np.nanmean(np.concatenate(pooled["sig_meanrm"]))) * 100.0
    sig_dt = float(np.mean(np.concatenate(pooled["sig_dt"]))) * 100.0
    print(f"\n=== POOLED over four pairs (calm within-window sigma: "
          f"mean-removed {sig_mean:.2f} cm, detrended {sig_dt:.2f} cm at 9 m) ===")
    print(f"  {'band':>5s} {'kappa':>7s} {'obs med |M|':>12s} "
          f"{'noise |M| meanrm':>17s} {'noise |M| detrend':>18s}")
    for band, lam in BANDS.items():
        k = kappa_of(lam)
        obs = np.nanmedian(np.concatenate(pooled["valid"][band]))
        n_mean = np.nanmedian(np.concatenate(pooled["noise_meanrm"][band]))
        n_dt = np.median(np.concatenate(pooled["noise_dt"][band]))
        print(f"  {band:>5s} {k:7.1f} {obs:12.3f} {n_mean:17.3f} {n_dt:18.3f}")


if __name__ == "__main__":
    main()
