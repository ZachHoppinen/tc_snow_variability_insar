"""noise_methods_compare.py -- compare three independent ASO depth-noise
estimates for the WY2025 pair, since the calm-region selection looks
unrepresentative (sparse for big events, terrain-biased, signal-contaminated).

  1. CALM region   -- detrended within-window sigma in smooth low-change snow
                      windows (the current method).
  2. PSD plateau   -- white-noise floor from the high-frequency tail of the 3 m
                      dSWE power spectrum. Real dSWE is red/fractal and rolls
                      off; ASO laser noise is white and plateaus. Uses the whole
                      snow field, no region selection.
  3. BARE GROUND   -- depth difference over pixels snow-free in BOTH flights
                      (depth < 10 cm both), where the true dSWE is identically
                      zero, so the spread is pure noise. A direct measurement.

All three are expressed as a per-9 m-cell noise sigma and the noise-only |M| it
implies, exp(-(kappa*sigma)^2/2), so they sit on the same axis as the observed
|M|. The PSD estimator is self-tested on synthetic white noise first.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ASO"))
sys.path.insert(0, str(HERE.parent))  # repo code/ dir
from aso_M_calculation import BANDS, PREAVG, kappa_of, load_dswe, preaverage  # noqa: E402
from aso_noise import calm_mask, coarse_window_fields, detrended_noise  # noqa: E402
from tc_paths import DATA_ROOT, FIG_ROOT  # noqa: E402

FIG_DIR = FIG_ROOT / "stats"
DSD_DIR = DATA_ROOT / ("sub_pixel_variability/"
                       "processed/aso/2025Feb08-09_2025Feb25/dsd")
DEPTH_A = DSD_DIR / "ASO_Tuolumne_2025Feb08-09_snowdepth_3m.tif"
DEPTH_B = DSD_DIR / "ASO_Tuolumne_2025Feb25_snowdepth_3m.tif"
PAIR = ("WY2025", "2025Feb08-09", "2025Feb25")
RHO = 0.373          # WY2025 snow density (g/cm3) -> depth(m) * RHO = SWE(m)
BARE_DEPTH_M = 0.10  # < 10 cm both flights = snow-free
TILE = 128           # 3 m PSD tile size (px)
N_TILES = 80


# --- PSD white-noise floor --------------------------------------------------
def radial_psd(tile):
    """Hann-windowed periodogram of a TILExTILE field, radially averaged.
    Normalized so a pure white field of variance s^2 returns a flat p with
    sigma^2 = N * p_floor (verified by psd_selftest)."""
    t = tile - np.mean(tile)
    w = np.outer(np.hanning(TILE), np.hanning(TILE))
    F = np.fft.fft2(w * t)
    P = (np.abs(F) ** 2) / (TILE * TILE * np.mean(w ** 2))   # ~Parseval-normalized
    ky = np.fft.fftfreq(TILE)[:, None]
    kx = np.fft.fftfreq(TILE)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)
    nb = TILE // 2
    edges = np.linspace(0, 0.5, nb + 1)
    idx = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, nb - 1)
    p = np.array([P.ravel()[idx == i].mean() for i in range(nb)])
    return edges[:-1], p


def psd_noise_sigma(field3m):
    """Per-pixel 3 m noise sigma from the high-k PSD plateau over finite tiles.
    Also returns the mean radial spectrum so the plateau can be inspected."""
    ny, nx = field3m.shape
    rng = np.random.default_rng(0)
    sig2, profs, kref = [], [], None
    tries = 0
    while len(sig2) < N_TILES and tries < N_TILES * 50:
        tries += 1
        r = rng.integers(0, ny - TILE)
        c = rng.integers(0, nx - TILE)
        t = field3m[r:r + TILE, c:c + TILE]
        if np.isfinite(t).mean() < 0.99:
            continue
        t = np.where(np.isfinite(t), t, np.nanmean(t))
        k, p = radial_psd(t)
        kref = k
        profs.append(p)
        floor = np.median(p[k > 0.35])              # outer 30% of frequencies
        sig2.append(floor)                          # sigma^2 = PSD plateau (this norm)
    return np.sqrt(np.median(sig2)), len(sig2), kref, np.mean(profs, axis=0)


def psd_selftest():
    """Recover a known white-noise sigma to validate the PSD normalization."""
    rng = np.random.default_rng(1)
    true = 0.02
    field = rng.normal(0, true, (512, 512))
    est, *_ = psd_noise_sigma(field)
    print(f"PSD self-test: injected sigma {true*100:.2f} cm -> "
          f"recovered {est*100:.2f} cm")


# --- bare-ground direct noise ----------------------------------------------
def bareground_noise_sigma():
    """Robust sigma of the depth difference over both-flights-snow-free pixels,
    converted to SWE. Per-pixel 3 m noise. Prints diagnostics because the depth
    product may be clamped to exactly 0 off-snow (which would kill the noise)."""
    a = rxr.open_rasterio(DEPTH_A, masked=True).squeeze().values
    b = rxr.open_rasterio(DEPTH_B, masked=True).squeeze().values
    bare = np.isfinite(a) & np.isfinite(b) & (a < BARE_DEPTH_M) & (b < BARE_DEPTH_M)
    diff = (b - a)[bare]                        # m depth, true change ~ 0
    frac_zero = np.mean(diff == 0.0)
    # restrict to thin-but-nonzero bare pixels to dodge clamped exact zeros
    nz = diff[diff != 0.0]
    print(f"  bare diag: n_bare={bare.sum():,}  frac exactly 0 = {frac_zero:.2f}  "
          f"a_min={np.nanmin(a):.3f} b_min={np.nanmin(b):.3f}")
    if nz.size:
        print(f"  bare nonzero diff (m): std={nz.std():.4f} "
              f"mad_sigma={1.4826*np.median(np.abs(nz-np.median(nz))):.4f} "
              f"p1/p99={np.percentile(nz,1):.3f}/{np.percentile(nz,99):.3f}")
    src = nz if nz.size > 1000 else diff
    src = src - np.median(src)
    sig_depth = 1.4826 * np.median(np.abs(src))
    return sig_depth * RHO, int(bare.sum())


def noiseM(sigma9, kappa):
    """Gaussian-model noise-only |M| from a per-cell noise sigma."""
    return np.exp(-0.5 * (kappa * sigma9) ** 2)


def main():
    wy, a, b = PAIR
    kappas = {bd: kappa_of(lam) for bd, lam in BANDS.items()}

    psd_selftest()

    # --- method 1: calm region (detrended), on the 9 m field
    d9 = preaverage(load_dswe(a, b))
    mean_d, _, cnt = coarse_window_fields(d9.values)
    calm = calm_mask(mean_d, cnt)
    dt_sig, _ = detrended_noise(d9.values, calm, kappas)
    sig9_calm = float(np.median(dt_sig))        # m SWE, already 9 m within-cell

    # --- method 2: PSD plateau on the 3 m field -> sigma_3m -> sigma_9m
    field3m = load_dswe(a, b).values
    sig3_psd, n_tiles, kref, prof = psd_noise_sigma(field3m)
    sig9_psd = sig3_psd / PREAVG                 # 3x3 average of iid white noise

    # --- method 3: bare ground -> sigma_3m -> sigma_9m
    sig3_bare, n_bare = bareground_noise_sigma()
    sig9_bare = sig3_bare / PREAVG

    rows = [("calm (detrend)", sig9_calm),
            ("PSD plateau", sig9_psd),
            ("bare ground", sig9_bare)]

    print(f"\n{wy}: per-9 m-cell noise sigma and implied noise-only |M|")
    print(f"  PSD tiles used: {n_tiles}   bare-ground pixels: {n_bare:,}")
    print(f"  {'method':>16s} {'sig9 (cm)':>10s} "
          f"{'|M| C':>7s} {'|M| L':>7s} {'|M| P':>7s}")
    for name, s9 in rows:
        mc = noiseM(s9, kappas['C'])
        ml = noiseM(s9, kappas['L'])
        mp = noiseM(s9, kappas['P'])
        print(f"  {name:>16s} {s9*100:10.2f} {mc:7.3f} {ml:7.3f} {mp:7.3f}")
    print(f"  {'OBSERVED |M|':>16s} {'':>10s} "
          f"{0.096:7.3f} {0.313:7.3f} {0.867:7.3f}   (WY2025, all windows)")

    # figure: (left) radial PSD plateau check, (right) noise-only |M| by method
    fig, (axp, ax) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axp.loglog(kref[1:], prof[1:], "k-")
    axp.axvspan(0.35, 0.5, color="orange", alpha=0.2, label="plateau band (noise)")
    axp.axhline(sig3_psd ** 2, color="red", ls="--",
                label=f"white floor -> sigma_3m={sig3_psd*100:.2f} cm")
    axp.set_xlabel("spatial frequency (cycles/px, 3 m)")
    axp.set_ylabel("radial PSD")
    axp.set_title(f"{wy}: 3 m dSWE radial spectrum\n(red signal rolls off, "
                  "white noise plateaus)")
    axp.legend(fontsize=8)

    names = [r[0] for r in rows]
    for band, col in (("C", "tab:blue"), ("L", "tab:green"), ("P", "tab:red")):
        ax.plot(names, [noiseM(s9, kappas[band]) for _, s9 in rows],
                "o-", color=col, label=f"{band} noise-only |M|")
        ax.axhline({"C": 0.096, "L": 0.313, "P": 0.867}[band], color=col,
                   ls=":", lw=1, label=f"{band} observed")
    ax.set_ylabel("|M|"); ax.set_ylim(0, 1)
    ax.set_title("noise-only |M| by estimation method")
    ax.legend(fontsize=8, ncol=3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "noise_methods_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
