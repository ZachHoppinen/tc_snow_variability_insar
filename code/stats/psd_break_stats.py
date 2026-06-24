"""psd_break_stats.py -- per-window two-regime spectral break of the native 3 m
dSWE field (Methods subsec:psd / the Figure~\\ref{fig:psd_break} numbers).

Source of truth for the break statistics quoted in the text. The figure script
(visualizations/plot_dswe_psd_break.py) imports compute_window_breaks() from
here and only plots, so the numbers and the figure stay in sync.

Method (Trujillo 2007 / Deems 2006). Tile the common snow mask into locally
homogeneous ~1.15 km windows on the native 3 m grid (kept at >=95% snow cover;
the <=5% invalid cells are filled with the window mean, i.e. zero after mean
removal, before the FFT). For each window and each ASO pair we form the
Hann-tapered radial periodogram in spatial wavenumber k, with NO detrend (Deems:
detrending biases the fractal slope toward short-range variability). Because the
four pairs have very different dSWE magnitudes, we AMPLITUDE-NORMALIZE each
(window, pair) spectrum -- center its log over the fit band, removing the
per-pair variance level -- before averaging the four pairs, so the shared SHAPE
is not dominated by the largest-accumulation pair. We fit a continuous two-
segment power law P(k) ~ k^-beta with a free break, taken as the segment
intersection. We also fit each pair separately to confirm the break is stable
across the four winters.

Wavenumber convention: P(k) ~ k^-beta, beta>0; the break is reported as a spatial
wavelength l_b = 1/k_b. The break scan floor (12 m = 2x the 3 m grid's 6 m
Nyquist wavelength) keeps fitted breaks clear of the noise-flattened near-Nyquist
band; we report the fraction of breaks landing near that floor as a check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  registers .rio
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ASO"))
sys.path.insert(0, str(HERE.parent))  # repo code/ dir
from aso_M_calculation import load_dswe, BASE_RES_M  # noqa: E402
from tc_paths import DATA_ROOT  # noqa: E402

PROCESSED_DIR = DATA_ROOT / "sub_pixel_variability/processed/aso"
MASK_3M = PROCESSED_DIR / "common_snow_mask_3m.tif"

PAIRS = [("WY2023", "2023Mar02-03", "2023Mar16-17"),
         ("WY2024", "2024Jan29", "2024Feb27-28"),
         ("WY2025", "2025Feb08-09", "2025Feb25"),
         ("WY2026", "2026Jan31-Feb01", "2026Feb27-28")]

TILE = 384             # 3 m cells -> 1152 m windows (= 128 cells at 9 m)
MIN_VALID = 0.95       # window kept if >= this snow-covered; rest window-mean filled
PXSIZE = BASE_RES_M    # 3 m  (grid Nyquist wavelength = 2*PXSIZE = 6 m)
FIT_WL = (9, 380)      # m, wavelengths in the fit (lower = 1.5x the 6 m Nyquist)
BREAK_SCAN = (12, 250)  # m, candidate break wavelengths (floor = 2x Nyquist)
MIN_R2 = 0.96
DEEMS_RANGE = (15, 40)  # cumulative snow-depth break range (Deems 2006)


def build_common_mask_3m():
    """Common snow mask = finite in ALL four pairs on the shared 3 m grid.
    Built once and cached; one field in memory at a time (~2.3 GB each)."""
    common, dims, coords = None, None, None
    for wy, a, b in PAIRS:
        da = load_dswe(a, b)
        v = np.isfinite(da.values)
        if dims is None:
            dims = da.dims
            coords = {key: da.coords[key].values for key in da.coords}
        common = v if common is None else (common & v)
        del da, v
    out = xr.DataArray(common.astype("uint8"), dims=dims, coords=coords)
    out.rio.write_crs("EPSG:32611", inplace=True)
    out.rio.to_raster(MASK_3M, compress="lzw")
    print(f"built {MASK_3M.name}  ({common.sum():,} common 3 m cells)")


def radial_psd(tile, T=TILE):
    """(k cycles/pixel, radially-averaged power, modes-per-bin) for a TxT field,
    mean removed. The <=5% invalid cells are window-mean filled by the caller.
    The mode count is the number of Fourier coefficients in each radial bin; it
    weights the fit so the few-mode innermost (large-scale) bins do not dominate."""
    t = tile - np.mean(tile)
    w = np.outer(np.hanning(T), np.hanning(T))
    F = np.fft.fft2(w * t)
    P = (np.abs(F) ** 2) / (T * T * np.mean(w ** 2))
    ky = np.fft.fftfreq(T)[:, None]
    kx = np.fft.fftfreq(T)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)
    nb = T // 2
    edges = np.linspace(0, 0.5, nb + 1)
    idx = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, nb - 1)
    p = np.array([P.ravel()[idx == i].mean() for i in range(nb)])
    counts = np.array([(idx == i).sum() for i in range(nb)])
    return edges[:-1], p, counts


def fit_break(wl, logp_or_p, weights, is_log=False):
    """Continuous two-segment fit of log P vs log(wavelength), by weighted least
    squares (weights = Fourier modes per radial bin). Returns
    (break_wl_m, beta_small_scale, beta_large_scale, R2, coef) or None. Pass a
    normalized log-spectrum with is_log=True, else a linear spectrum."""
    y_all = logp_or_p if is_log else np.log(np.where(logp_or_p > 0, logp_or_p, np.nan))
    use = (wl >= FIT_WL[0]) & (wl <= FIT_WL[1]) & np.isfinite(y_all)
    if use.sum() < 10:
        return None
    x = np.log(wl[use]); y = y_all[use]
    sw = np.sqrt(weights[use])                         # row scaling for WLS
    wmean = np.sum(weights[use] * y) / np.sum(weights[use])
    sst = np.sum(weights[use] * (y - wmean) ** 2)
    best = None
    for lam in np.linspace(*BREAK_SCAN, 50):
        xb = np.log(lam)
        if (x < xb).sum() < 3 or (x > xb).sum() < 3:
            continue
        B = np.column_stack([np.ones_like(x), x, np.clip(x - xb, 0, None)])
        coef, *_ = np.linalg.lstsq(B * sw[:, None], y * sw, rcond=None)
        sse = np.sum(weights[use] * (y - B @ coef) ** 2)
        if best is None or sse < best[0]:
            best = (sse, lam, coef)
    if best is None:
        return None
    sse, lam, coef = best
    r2 = 1 - sse / sst
    return lam, coef[1], coef[1] + coef[2], r2, coef


def compute_window_breaks():
    """Per-window break fits (amplitude-normalized, 4-pair mean) plus per-pair
    fits, in one pass. Returns a dict used by this script (stats) and the figure
    script (plotting)."""
    if not MASK_3M.exists():
        build_common_mask_3m()
    common = rioxarray.open_rasterio(MASK_3M).squeeze(drop=True).values.astype(bool)
    ny, nx = common.shape
    tiles = [(r, c)
             for r in range(0, ny - TILE + 1, TILE)
             for c in range(0, nx - TILE + 1, TILE)
             if common[r:r + TILE, c:c + TILE].mean() >= MIN_VALID]
    if not tiles:
        raise RuntimeError("no windows; lower MIN_VALID or TILE")
    n_win = len(tiles)
    wynames = [p[0] for p in PAIRS]

    # raw per-pair per-window PSD (one field in memory at a time)
    psd_pair, k, counts = {}, None, None
    for wy, a, b in PAIRS:
        f = load_dswe(a, b).values
        arr = None
        for i, (r, c) in enumerate(tiles):
            t = f[r:r + TILE, c:c + TILE]
            if not np.isfinite(t).all():                # <=5% gaps -> window mean
                t = np.where(np.isfinite(t), t, np.nanmean(t))
            kk, p, cnts = radial_psd(t)
            if arr is None:
                k = kk; counts = cnts; arr = np.zeros((n_win, len(p)))
            arr[i] = p
        psd_pair[wy] = arr
        del f
    wl = 1.0 / (k[1:] / PXSIZE)
    use = (wl >= FIT_WL[0]) & (wl <= FIT_WL[1])
    wcnt = counts[1:].astype(float)                    # modes per bin (drop k=0)

    # amplitude-normalize each (pair, window): center log over the fit band
    def norm_log(p_full):
        lp = np.log(p_full[1:])
        lp -= lp[use].mean()
        return lp

    spec_pw = {wy: np.full((n_win, wl.size), np.nan) for wy in wynames}
    valid = []
    for i in range(n_win):
        if all(np.all(np.isfinite(psd_pair[wy][i][1:][use]))
               and np.all(psd_pair[wy][i][1:][use] > 0) for wy in wynames):
            valid.append(i)
            for wy in wynames:
                spec_pw[wy][i] = norm_log(psd_pair[wy][i])
    valid = np.array(valid)

    # per-window mean over the four (normalized) pairs -> fit -> per-window break
    breaks, bsmall, blarge, win_specs = [], [], [], []
    for i in valid:
        wspec = np.mean([spec_pw[wy][i] for wy in wynames], axis=0)
        win_specs.append(wspec)
        res = fit_break(wl, wspec, wcnt, is_log=True)
        if res is None:
            continue
        lam, bs, bl, r2, _ = res
        if r2 < MIN_R2 or lam <= BREAK_SCAN[0] + 1 or lam >= BREAK_SCAN[1] - 1:
            continue
        breaks.append(lam); bsmall.append(bs); blarge.append(bl)
    win_specs = np.array(win_specs)

    # per-pair mean spectrum over valid windows -> fit -> per-pair break (4 winters)
    pair_fit = {}
    for wy in wynames:
        pspec = np.nanmean(spec_pw[wy][valid], axis=0)
        pair_fit[wy] = fit_break(wl, pspec, wcnt, is_log=True)

    # overall mean spectrum (figure mean line + slope lines)
    mean_lp = win_specs.mean(axis=0)
    mlam, mbs, mbl, _, mcoef = fit_break(wl, mean_lp, wcnt, is_log=True)

    return dict(n_total=n_win, n_valid=valid.size,
                breaks=np.array(breaks), bsmall=np.array(bsmall),
                blarge=np.array(blarge), wl=wl, use=use, specs=win_specs,
                mean_lp=mean_lp, mlam=mlam, mbs=mbs, mbl=mbl, mcoef=mcoef,
                pair_fit=pair_fit)


def main():
    R = compute_window_breaks()
    br = R["breaks"]; n = br.size
    in_deems = np.mean((br >= DEEMS_RANGE[0]) & (br <= DEEMS_RANGE[1]))
    near_floor = np.mean(br <= BREAK_SCAN[0] + 3)
    print("Per-window 3 m dSWE PSD two-regime break "
          "(amplitude-normalized, 4-pair mean)\n")
    print(f"  windows passing R2>={MIN_R2}, interior break : {n}/{R['n_total']}")
    print(f"  break wavelength l_b : median {np.median(br):.0f} m  "
          f"IQR [{np.percentile(br,25):.0f}, {np.percentile(br,75):.0f}]")
    print(f"  small-scale beta     : median {np.median(R['bsmall']):.2f}")
    print(f"  large-scale beta     : median {np.median(R['blarge']):.2f}")
    print(f"  within Deems {DEEMS_RANGE[0]}-{DEEMS_RANGE[1]} m : {100*in_deems:.0f}%")
    print(f"  within 3 m of scan floor ({BREAK_SCAN[0]} m) : {100*near_floor:.0f}%  "
          "(should be small -- no pile-up at the floor)")
    print("\n  per-pair break (stability across the four winters):")
    for wy, *_ in PAIRS:
        f = R["pair_fit"][wy]
        if f is None:
            print(f"    {wy}: fit failed")
        else:
            lam, bs, bl, r2, _ = f
            print(f"    {wy}: l_b={lam:.0f} m  beta {bs:.2f}/{bl:.2f}  (R2={r2:.3f})")


if __name__ == "__main__":
    main()
