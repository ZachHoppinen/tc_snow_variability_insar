"""stats_constant_density.py -- does the basin-constant snow density change |M|
or arg(M) versus a spatially-varying density?

The pipeline converts 3 m ASO depth differences to dSWE with one basin-constant
density per pair. ASO's own products carry a spatially-varying density model.
We rebuild a variable-density dSWE field and compare its |M|/arg(M) to the
constant-density field on the same 81 m windows.

To isolate the SPATIAL density variation (not a mean offset), the variable
field is scaled to the same basin-mean density as the constant field:

  rho_var(x,y) = dSWE_50m / ddepth_50m            (ASO's per-cell density)
  ratio        = rho_var / mean(rho_var)          (mean 1, spatial variation only)
  dswe_var_3m  = dswe_const_3m * upsample(ratio)

Both fields then go through the identical 3 m -> 9 m -> 81 m M pipeline. If the
constant-density assumption is harmless, |M| and arg(M) track 1:1.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import xarray as xr  # noqa: F401
from rasterio.enums import Resampling

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ASO"))
sys.path.insert(0, str(HERE.parent))  # repo code/ dir
from aso_M_calculation import (  # noqa: E402
    BANDS, PAIRS, PROCESSED_DIR, kappa_of, load_dswe, preaverage, window_M,
)
from tc_paths import FIG_ROOT  # noqa: E402

FIG_DIR = FIG_ROOT / "stats"
KAPPA = kappa_of(BANDS["L"])        # compare at L-band, the analysis band


def p50(pair_dir, date, kind, sub):
    return PROCESSED_DIR / pair_dir / sub / f"ASO_Tuolumne_{date}_{kind}_50m.tif"


def density_ratio(early, late):
    """Per-50 m-cell density ratio rho_var/mean(rho_var) as a DataArray, from the
    PER-FLIGHT snow density SWE/depth (a stable, smooth field) averaged over the
    two flights. Invalid cells -> ratio 1 (use the constant)."""
    pd = f"{early}_{late}"
    swe_a = rxr.open_rasterio(p50(pd, early, "swe", "dswe"), masked=True).squeeze()
    swe_b = rxr.open_rasterio(p50(pd, late, "swe", "dswe"), masked=True).squeeze()
    dep_a = rxr.open_rasterio(p50(pd, early, "snowdepth", "dsd"), masked=True).squeeze()
    dep_b = rxr.open_rasterio(p50(pd, late, "snowdepth", "dsd"), masked=True).squeeze()
    if dep_a.shape != swe_a.shape:
        dep_a = dep_a.rio.reproject_match(swe_a)
        dep_b = dep_b.rio.reproject_match(swe_a)

    def flight_rho(swe, dep):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = swe.values / dep.values            # per-flight density (fraction)
        ok = (dep.values >= 0.30) & (r >= 0.10) & (r <= 0.60)
        return np.where(ok, r, np.nan)

    rho = np.nanmean([flight_rho(swe_a, dep_a), flight_rho(swe_b, dep_b)], axis=0)
    rho_mean = np.nanmean(rho)
    ratio = np.where(np.isfinite(rho), rho / rho_mean, 1.0)   # mean ~1
    out = swe_a.copy(data=ratio.astype("float32"))
    return out, rho_mean


def pair_compare(early, late):
    """Per-window |M| and arg(M) for constant- and variable-density fields."""
    dswe_const = load_dswe(early, late)                       # 3 m, constant rho
    ratio50, rho_mean = density_ratio(early, late)
    ratio3 = ratio50.rio.reproject_match(dswe_const, resampling=Resampling.nearest)
    dswe_var = dswe_const * ratio3                            # 3 m, variable rho

    M_const = window_M(preaverage(dswe_const), KAPPA).values
    M_var = window_M(preaverage(dswe_var), KAPPA).values
    ok = np.isfinite(M_const) & np.isfinite(M_var)
    return (np.abs(M_const)[ok], np.abs(M_var)[ok],
            np.degrees(np.angle(M_const))[ok], np.degrees(np.angle(M_var))[ok],
            rho_mean)


def main():
    print(f"L-band |M| and arg(M): constant vs spatially-varying density\n")
    print(f"{'pair':8s} {'rho_mean':>8s} {'|M|c':>6s} {'|M|v':>6s} "
          f"{'d|M|':>7s} {'r|M|':>6s} {'argc':>7s} {'argv':>7s} "
          f"{'darg':>7s} {'r_arg':>6s}")
    fig_data = None
    for wy, a, b in PAIRS:
        mc, mv, ac, av, rho = pair_compare(a, b)
        dM = np.median(np.abs(mv - mc))
        rM = np.corrcoef(mc, mv)[0, 1]
        dA = np.median(np.abs(av - ac))
        rA = np.corrcoef(ac, av)[0, 1]
        print(f"{wy:8s} {rho:8.3f} {np.median(mc):6.3f} {np.median(mv):6.3f} "
              f"{dM:7.4f} {rM:6.3f} {np.median(ac):+7.2f} {np.median(av):+7.2f} "
              f"{dA:7.3f} {rA:6.3f}")
        if wy == "WY2025":
            fig_data = (mc, mv, ac, av)

    # figure: const vs var for WY2025, |M| and arg(M)
    mc, mv, ac, av = fig_data
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax0.hexbin(mc, mv, gridsize=80, bins="log", cmap="viridis", extent=(0, 1, 0, 1))
    ax0.plot([0, 1], [0, 1], "r--", lw=1)
    ax0.set_xlabel("|M| constant density"); ax0.set_ylabel("|M| variable density")
    ax0.set_title("WY2025: |M| constant vs variable density")
    ax1.hexbin(ac, av, gridsize=80, bins="log", cmap="viridis",
               extent=(-90, 90, -90, 90))
    ax1.plot([-90, 90], [-90, 90], "r--", lw=1)
    ax1.set_xlabel("arg(M) constant (deg)"); ax1.set_ylabel("arg(M) variable (deg)")
    ax1.set_title("WY2025: arg(M) constant vs variable density")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "constant_vs_variable_density.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
