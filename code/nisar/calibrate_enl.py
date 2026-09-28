"""calibrate_enl.py -- effective number of looks (ENL) over water, via the
full Bamler multilook-coherence bias model.

Over open water the true coherence is ~0--small, so the observed coherence is
dominated by estimator bias. The oversampling factor f (sample correlation) is a
single property of the instrument/processing, so ENL = N_looks / f must give ONE
f for both resolutions. We find it by a JOINT DIFFERENTIAL calibration over
water (following multi_resolution/water_shadow_bias_check.py): for a candidate f,
de-bias the 80 m coherence to a true coherence, predict the 20 m coherence under
the same f, and require the prediction to match the observed 20 m coherence.
This is insensitive to the residual real water coherence (it enters as the true
coherence), unlike inverting each resolution's gamma=0 floor independently.

Bamler & Hartl (1998) / Touzi et al. (1999) expected sample coherence:
    E{gamma_hat | gamma, L}
      = Gamma(L)Gamma(3/2)/Gamma(L+1/2)
        * 3F2(3/2,L,L; L+1/2,1; gamma^2) * (1-gamma^2)^L

The de-biasers built here (one per L) are reused downstream to correct the
20 m and 80 m coherence in the |M| analysis.

Layers (delivered coherenceMagnitude), pooled over the two 12-day legs P12, P23:
    wrappedInterferogram    20 m   5x6  = 30  nominal looks
    unwrappedInterferogram  80 m   13x16= 208 nominal looks
"""

from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  registers .rio
import mpmath as mp
from rasterio.enums import Resampling
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo code/ dir
from tc_paths import DATA_ROOT  # noqa: E402

# ============================================================================
# the sub-pixel-variability data tree (override with TC_DATA_ROOT).
# ============================================================================
RAW_MAIN = DATA_ROOT / "sub_pixel_variability"

GUNW_DIR = RAW_MAIN / "raw/nisar/gunw"
# the two delivered standard GUNWs, by their original ASF product names
# (downloaded by nisar/download/fetch_paper_gunws.py); Provisional release P05023 (v1.0.11), was Beta X05010
GUNWS = [GUNW_DIR / ("NISAR_L2_PR_GUNW_007_042_D_069_008_4000_SH_20251207T025607_"
                     "20251207T025642_20251219T025608_20251219T025642_P05023_N_F_J_001.h5"),
         GUNW_DIR / ("NISAR_L2_PR_GUNW_008_042_D_069_009_4000_SH_20251219T025608_"
                     "20251219T025642_20251231T025608_20251231T025643_P05023_N_F_J_001.h5")]
WATER_MASK = (Path(__file__).resolve().parent.parent
              / "cache/tuolome_water_mask_worldcover_80m.tif")
COH_MAX = 0.20                       # open-water cut on the 80 m coherence
N_LOOKS_20, N_LOOKS_80 = 5 * 6, 13 * 16


# ---------------------------------------------------------------------------
# Bamler coherence-bias model + reusable de-biaser (used here and downstream).
# ---------------------------------------------------------------------------
def bamler_mean(gamma: float, L: float) -> float:
    """E{gamma_hat | gamma, L}."""
    if gamma >= 1.0:
        return 1.0
    g2 = gamma * gamma
    pref = mp.gamma(L) * mp.gamma(1.5) / mp.gamma(L + 0.5)
    return float(pref * mp.hyp3f2(1.5, L, L, L + 0.5, 1, g2) * (1 - g2) ** L)


def forward_curve(L: float, n: int = 120, gmax: float = 0.999):
    """Sampled (gamma, E{gamma_hat|gamma,L}) curve, monotone in gamma. gmax<1
    keeps it fast/stable when only low coherence is needed (e.g. water)."""
    g = np.linspace(0.0, gmax, n)
    if gmax >= 0.999:
        g = np.append(g, 1.0)
    ghat = np.array([bamler_mean(gi, L) for gi in g])
    return g, ghat


def make_debiaser(L: float, n: int = 200):
    """Return ghat -> gamma_true map for looks L (clamped to [0,1])."""
    g, ghat = forward_curve(L, n)
    return interp1d(ghat, g, bounds_error=False, fill_value=(0.0, 1.0))


# ---------------------------------------------------------------------------
def read_coh(gunw: Path, group: str) -> xr.DataArray:
    base = f"science/LSAR/GUNW/grids/frequencyA/{group}"
    with h5py.File(gunw, "r") as f:
        coh = f[f"{base}/HH/coherenceMagnitude"][:].astype(np.float32)
        x = f[f"{base}/xCoordinates"][:]; y = f[f"{base}/yCoordinates"][:]
        epsg = int(f[f"{base}/projection"][()])
    da = xr.DataArray(coh, dims=("y", "x"), coords={"y": y, "x": x}, name="coh")
    da.rio.write_crs(f"EPSG:{epsg}", inplace=True)
    return da


def water_pairs():
    """Paired (gamma_hat_20, gamma_hat_80) at open-water pixels, pooled over legs."""
    mask = rioxarray.open_rasterio(WATER_MASK).squeeze("band", drop=True)
    g20, g80 = [], []
    for gunw in GUNWS:
        c80 = read_coh(gunw, "unwrappedInterferogram")
        c20 = read_coh(gunw, "wrappedInterferogram").rio.reproject_match(
            c80, resampling=Resampling.average)            # 20 m coh averaged to 80 m grid
        m = mask.rio.reproject_match(c80, resampling=Resampling.nearest).values == 1
        w = m & np.isfinite(c80.values) & np.isfinite(c20.values) & (c80.values < COH_MAX)
        g80.append(c80.values[w]); g20.append(c20.values[w])
    return np.concatenate(g20), np.concatenate(g80)


def main():
    g20, g80 = water_pairs()
    print(f"open-water pixels (pooled P12+P23): {g20.size}")
    print(f"  median gamma_hat: 20 m = {np.median(g20):.4f}, 80 m = {np.median(g80):.4f}")

    def median_residual(f: float) -> float:
        """Median (obs_20 - predicted_20) at water, predicting 20 m from the
        de-biased 80 m under oversampling factor f. Zero at the correct f.
        Water coherence is small, so a low-gamma curve suffices (fast/stable)."""
        gg20, e20 = forward_curve(N_LOOKS_20 / f, n=80, gmax=0.5)
        gg80, e80 = forward_curve(N_LOOKS_80 / f, n=80, gmax=0.5)
        gtrue = np.interp(g80, e80, gg80)        # de-bias 80 m -> true coherence
        pred20 = np.interp(gtrue, gg20, e20)     # predict 20 m under same f
        return float(np.median(g20 - pred20))

    # single oversampling factor where the water residual vanishes
    f_star = brentq(median_residual, 1.0, 4.0, xtol=1e-3)
    enl20, enl80 = N_LOOKS_20 / f_star, N_LOOKS_80 / f_star
    print(f"\njoint water calibration:")
    print(f"  oversampling factor f = {f_star:.3f}  (single value, both resolutions)")
    print(f"  ENL_20 = {N_LOOKS_20}/f = {enl20:.1f}")
    print(f"  ENL_80 = {N_LOOKS_80}/f = {enl80:.1f}")
    print(f"  residual at f* = {median_residual(f_star):+.4f} (should be ~0)")


if __name__ == "__main__":
    main()
