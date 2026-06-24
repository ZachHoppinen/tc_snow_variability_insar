"""corrected_coherence_p23.py -- bias-corrected 20-80 m coherence difference for
the snowy P23 pair (2025-12-19 -> 2025-12-31), Tuolumne.

The delivered GUNW carries two coherence layers at different looks:
    wrappedInterferogram     20 m   5x6  = 30  nominal looks  (L_20 = 30 / f)
    unwrappedInterferogram   80 m   13x16= 208 nominal looks  (L_80 = 208 / f)
With a single oversampling factor f = 1.287 (calibrate_enl.py) the effective
looks are L_20 ~ 23.3 and L_80 ~ 161.6.

A finite number of looks biases the coherence estimate UP, and more so at fewer
looks, so the raw 20-80 m gap is partly just this multilook-bias gap. We remove
it by de-biasing EACH resolution independently -- inverting the Bamler curve at
its own L (this needs only that pixel's sample coherence and its looks) -- and
then differencing the corrected coherences:

    dgamma = gamma_tilde_20 - gamma_tilde_80          (sub-pixel coherence loss)

Steps (each printed/plotted so bad assumptions surface early):
  1. read both delivered coherence layers
  2. block-average the 20 m layer onto the common 80 m grid
  3. build the two de-biasers and de-bias each layer
  4. difference, with water as a built-in sanity check (calibration forced
     corrected dgamma ~ 0 over water)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rioxarray  # noqa: F401  registers .rio
from rasterio.enums import Resampling
from scipy.interpolate import interp1d

# reuse the GUNW reader and the Bamler forward model from the calibration script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_enl import read_coh, forward_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo code/ dir
from tc_paths import DATA_ROOT, FIG_ROOT  # noqa: E402

# ============================================================================
# inputs + the single calibrated oversampling factor
# ============================================================================
RAW_MAIN = DATA_ROOT / "sub_pixel_variability"
GUNW_DIR = RAW_MAIN / "raw/nisar/gunw"
PAIRS = {                                  # the two delivered standard GUNWs,
                                           # by their original ASF product names
    "P12": ("NISAR_L2_PR_GUNW_007_042_D_069_008_4000_SH_20251207T025607_"
            "20251207T025642_20251219T025608_20251219T025642_X05010_N_F_J_001.h5",
            "2025-12-07 -> 2025-12-19"),
    "P23": ("NISAR_L2_PR_GUNW_008_042_D_069_009_4000_SH_20251219T025608_"
            "20251219T025642_20251231T025608_20251231T025643_X05010_N_F_J_001.h5",
            "2025-12-19 -> 2025-12-31"),
}
WATER_MASK = (Path(__file__).resolve().parent.parent
              / "cache/tuolome_water_mask_worldcover_80m.tif")

F_OVERSAMPLE = 1.287                      # single value, both resolutions (calibrate_enl.py)
N_LOOKS_20, N_LOOKS_80 = 5 * 6, 13 * 16
L_20, L_80 = N_LOOKS_20 / F_OVERSAMPLE, N_LOOKS_80 / F_OVERSAMPLE

FIG_DIR = FIG_ROOT / "etc"
COH_OUT = RAW_MAIN / "processed/nisar/coherence"     # raw + corrected GeoTIFFs


def correct_native(coh, deb):
    """De-bias a coherence layer on its OWN grid (per-pixel inversion needs only
    the sample coherence and its looks). NaNs are preserved."""
    v = coh.values
    out = np.full_like(v, np.nan)
    m = np.isfinite(v)
    out[m] = deb(v[m])
    return coh.copy(data=out)


def debiaser(L: float, gmax: float, n: int = 140):
    """Map observed sample coherence -> de-biased true coherence at looks L, by
    inverting the (monotone) Bamler mean curve. gmax caps the curve below 1 to
    keep the mpmath _3F_2 fast/stable; observed coherence above E{ghat|gmax}
    clamps to 1, below the gamma=0 floor clamps to 0."""
    g, ghat = forward_curve(L, n=n, gmax=gmax)
    return interp1d(ghat, g, bounds_error=False, fill_value=(0.0, 1.0))


def main():
    ap = argparse.ArgumentParser(description="corrected 20-80 m coherence for one pair")
    ap.add_argument("--pair", choices=sorted(PAIRS), default="P23")
    pair = ap.parse_args().pair
    gunw_name, date_label = PAIRS[pair]
    gunw = GUNW_DIR / gunw_name
    print(f"pair {pair}  ({date_label})  {gunw_name}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- step 1: read the two delivered coherence layers --------------------
    c20 = read_coh(gunw, "wrappedInterferogram")      # 20 m, 5x6 looks
    c80 = read_coh(gunw, "unwrappedInterferogram")    # 80 m, 13x16 looks
    print(f"step 1  raw coherence  20 m grid {c20.shape}, 80 m grid {c80.shape}")
    print(f"        median raw gamma_hat:  20 m = {np.nanmedian(c20.values):.3f}, "
          f"80 m = {np.nanmedian(c80.values):.3f}")

    # --- step 2: block-average the 20 m layer onto the 80 m grid ------------
    # (average-then-debias matches the water calibration; the mean of ~16 20 m
    #  coherences still carries the L_20 bias, so we de-bias it with L_20.)
    c20_on80 = c20.rio.reproject_match(c80, resampling=Resampling.average)
    g20 = c20_on80.values
    g80 = c80.values
    valid = np.isfinite(g20) & np.isfinite(g80)
    print(f"step 2  averaged 20 m -> 80 m grid {c20_on80.shape}; "
          f"{valid.sum()} valid pixels")

    # --- step 3: build de-biasers and correct each resolution ---------------
    # snow coherence reaches higher than water, so use a wider gmax than the
    # water calibration; 20 m has the bigger floor (correction matters most there).
    print(f"step 3  building de-biasers  (L_20 = {L_20:.1f}, L_80 = {L_80:.1f}) ...")
    deb20 = debiaser(L_20, gmax=0.97)
    deb80 = debiaser(L_80, gmax=0.92)
    print(f"        gamma=0 floors:  20 m ~ {0.886/np.sqrt(L_20):.3f}, "
          f"80 m ~ {0.886/np.sqrt(L_80):.3f}  (raw below the floor -> 0)")

    gt20 = np.full_like(g20, np.nan)
    gt80 = np.full_like(g80, np.nan)
    gt20[valid] = deb20(g20[valid])
    gt80[valid] = deb80(g80[valid])

    dgamma_raw = np.where(valid, g20 - g80, np.nan)        # uncorrected gap
    dgamma = np.where(valid, gt20 - gt80, np.nan)          # corrected loss

    # --- step 3b: save raw + corrected coherence GeoTIFFs (native posting) ---
    # de-bias each layer on its OWN grid (no averaging) for standalone products.
    COH_OUT.mkdir(parents=True, exist_ok=True)
    products = [
        (c20, f"{pair}_coherence_raw_20m.tif"),
        (c80, f"{pair}_coherence_raw_80m.tif"),
        (correct_native(c20, deb20), f"{pair}_coherence_corrected_20m.tif"),
        (correct_native(c80, deb80), f"{pair}_coherence_corrected_80m.tif"),
    ]
    for da, name in products:
        da.rio.to_raster(COH_OUT / name)
        print(f"        wrote {name}")

    # --- step 4: water sanity check + stats ---------------------------------
    wm = (rioxarray.open_rasterio(WATER_MASK).squeeze("band", drop=True)
          .rio.reproject_match(c80, resampling=Resampling.nearest).values == 1)
    water = valid & wm
    land = valid & ~wm
    print(f"step 4  corrected dgamma medians:")
    print(f"          water (should be ~0): {np.nanmedian(dgamma[water]):+.3f}  "
          f"(n={water.sum()})")
    print(f"          land  (snow signal):  {np.nanmedian(dgamma[land]):+.3f}  "
          f"(n={land.sum()})")
    print(f"        raw (uncorrected) land gap: {np.nanmedian(dgamma_raw[land]):+.3f}")

    # ---- figure: raw (top) vs corrected (bottom) coherence + difference ----
    ext = [float(c80.x.min()), float(c80.x.max()),
           float(c80.y.min()), float(c80.y.max())]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    ckw = dict(cmap="viridis", vmin=0, vmax=1, extent=ext, origin="upper")
    dkw = dict(cmap="RdBu_r", vmin=-0.3, vmax=0.3, extent=ext, origin="upper")

    panels = [
        (ax[0, 0], g20, "raw $\\hat\\gamma_{20}$", ckw),
        (ax[0, 1], g80, "raw $\\hat\\gamma_{80}$", ckw),
        (ax[0, 2], dgamma_raw, "raw $\\hat\\gamma_{20}-\\hat\\gamma_{80}$", dkw),
        (ax[1, 0], gt20, "corrected $\\tilde\\gamma_{20}$", ckw),
        (ax[1, 1], gt80, "corrected $\\tilde\\gamma_{80}$", ckw),
        (ax[1, 2], dgamma, "corrected $\\Delta\\gamma=\\tilde\\gamma_{20}-\\tilde\\gamma_{80}$", dkw),
    ]
    for a, data, title, kw in panels:
        im = a.imshow(data, **kw)
        a.set_title(title, fontsize=12)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle(f"{pair}  {date_label}   (f = "
                 f"{F_OVERSAMPLE}, $L_{{20}}$={L_20:.0f}, $L_{{80}}$={L_80:.0f})",
                 fontsize=14)
    out = FIG_DIR / f"corrected_coherence_{pair.lower()}_maps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote: {out}")

    # ---- figure: corrected dgamma histogram, water vs land -----------------
    fig, a = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bins = np.linspace(-0.3, 0.5, 80)
    a.hist(dgamma[water], bins=bins, density=True, alpha=0.6,
           color="tab:blue", label=f"water (median {np.nanmedian(dgamma[water]):+.3f})")
    a.hist(dgamma[land], bins=bins, density=True, alpha=0.6,
           color="tab:green", label=f"land/snow (median {np.nanmedian(dgamma[land]):+.3f})")
    a.axvline(0, color="0.4", lw=1)
    a.set_xlabel(r"corrected $\Delta\gamma = \tilde\gamma_{20} - \tilde\gamma_{80}$")
    a.set_ylabel("density")
    a.set_title(f"{pair} sub-pixel coherence loss (water ~0 = calibration check)")
    a.legend()
    out = FIG_DIR / f"corrected_coherence_{pair.lower()}_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
