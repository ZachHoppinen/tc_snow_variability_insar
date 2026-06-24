"""Wrapped interferometric phase vs Delta-SWE at C, L, and P band (v4).

Forward Leinss model: each band's phase grows linearly with Delta-SWE at the
rate kappa(lambda, theta) and wraps through +/-pi. We overlay, as error bars,
the basin-mean Delta-SWE +/- sigma for the four Tuolumne ASO pairs, computed
LIVE from the v4 data tree (no hardcoded stats), so the figure stays in sync
with whatever pairs the pipeline uses.

The plotted Delta-SWE is ASO's own 50 m Delta-SWE (swe_B - swe_A), the measured
SWE change, over snow-bearing cells (>= 10 cm depth in either flight).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)

# --- paths ------------------------------------------------------------------
from plotting_constants import (  # noqa: E402
    apply_style, DPI, PROCESSED_DIR, FIG_DIR, PAIRS, PAIR_COLORS, BAND_COLORS)
BASIN = "Tuolumne"

# (water-year label, early flight, late flight) -- same pairs as the pipeline.

# --- physics ----------------------------------------------------------------
INC_RAD = np.deg2rad(40.0)
DSWE_MAX_CM = 50.0
N_SAMPLES = 10001
MIN_DEPTH_M = 0.10                  # snow-bearing mask threshold
# (band, wavelength m, color, linewidth, alpha) -- C is thinner/lighter as its
# many wraps over 50 cm crowd the panel.
BANDS = [
    ("C-band ($\\lambda$ = 5.6 cm)",  0.056, BAND_COLORS["C"], 1.2, 0.8),
    ("L-band ($\\lambda$ = 23.8 cm)", 0.238, BAND_COLORS["L"], 2.2, 1.0),
    ("P-band ($\\lambda$ = 81 cm)",   0.81,  BAND_COLORS["P"], 2.2, 1.0),
]


def kappa(wavelength_m: float) -> float:
    """Leinss phase-per-SWE factor [rad / m SWE] at the 40 deg incidence."""
    return (2 * np.pi / wavelength_m) * (1.59 + INC_RAD ** 2.5)


def open_raster(path: Path):
    """Open a single-band raster fully into memory, NaN nodata, handle closed."""
    with rioxarray.open_rasterio(path, masked=True) as da:
        return da.squeeze(drop=True).load()


def pair_dswe50_stats(date_a: str, date_b: str) -> tuple[float, float]:
    """Basin mean and std (cm) of the 50 m ASO Delta-SWE (swe_B - swe_A) over
    snow-bearing cells, for one pair. All 50 m products share the grid, so the
    difference is a direct subtraction."""
    pdir = PROCESSED_DIR / f"{date_a}_{date_b}"
    swe_a = open_raster(pdir / "dswe" / f"ASO_{BASIN}_{date_a}_swe_50m.tif")
    swe_b = open_raster(pdir / "dswe" / f"ASO_{BASIN}_{date_b}_swe_50m.tif")
    sd_a = open_raster(pdir / "dsd" / f"ASO_{BASIN}_{date_a}_snowdepth_50m.tif")
    sd_b = open_raster(pdir / "dsd" / f"ASO_{BASIN}_{date_b}_snowdepth_50m.tif")

    dswe_cm = (swe_b.values - swe_a.values) * 100.0
    snow = (sd_a.values >= MIN_DEPTH_M) | (sd_b.values >= MIN_DEPTH_M)
    vals = dswe_cm[snow & np.isfinite(dswe_cm)]
    return float(vals.mean()), float(vals.std())


def break_at_wraps(phi: np.ndarray) -> np.ndarray:
    """Insert NaN at each |jump| > pi so matplotlib doesn't draw the wrap line."""
    out = phi.copy().astype(np.float64)
    out[1:][np.abs(np.diff(phi)) > np.pi] = np.nan
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    dswe_cm = np.linspace(0.0, DSWE_MAX_CM, N_SAMPLES)
    dswe_m = dswe_cm / 100.0

    # per-pair basin stats (live from the data), kept in water-year order (WY2023->WY2026)
    stats = [(wy, *pair_dswe50_stats(da, db)) for wy, da, db in PAIRS]
    print("pair 50 m ASO dSWE (snow cells):")
    for wy, m, s in stats:
        print(f"  {wy}: mu = {m:5.1f} cm, sigma = {s:5.1f} cm")

    print("\nwrap rates at 40 deg:")
    for label, wl, *_ in BANDS:
        k = kappa(wl)
        print(f"  {label:<28s} kappa = {k:6.1f} rad/m, "
              f"{2 * np.pi / k * 100:5.2f} cm/wrap")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    # band wrap curves, labeled by the distance between wraps (directly
    # comparable to the pair error bars above)
    for label, wl, color, lw, alpha in BANDS:
        k = kappa(wl)
        phi = k * dswe_m
        cm_per_wrap = 2 * np.pi / k * 100
        ax.plot(dswe_cm, break_at_wraps(np.angle(np.exp(1j * phi))),
                color=color, lw=lw, alpha=alpha,
                label=rf"{label}: wraps every {cm_per_wrap:.1f} cm")

    # phase-axis decoration
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    for yl in (np.pi, -np.pi):
        ax.axhline(yl, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    # per-pair mu +/- sigma strip above the data, labeled directly on each bar
    # (no legend entries) so the strip is self-explanatory
    n = len(stats)
    levels = np.linspace(np.pi + 0.55, np.pi + 0.55 + 0.5 * (n - 1), n)
    for (wy, m, s), y in zip(stats, levels[::-1]):   # WY2023 at top -> WY2026 below
        ax.errorbar(m, y, xerr=s, fmt="o", color=PAIR_COLORS[wy], markersize=8,
                    capsize=6, lw=2.2, alpha=0.95, zorder=4)
        ax.annotate(rf"{wy}: {m:.0f}$\,\pm\,${s:.0f} cm", (m, y),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                    color=PAIR_COLORS[wy], zorder=5)
    ax.axhline(np.pi + 0.25, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.text(DSWE_MAX_CM - 0.5, np.pi + 0.32, "basin $\\Delta$SWE per lidar pair",
            ha="right", va="bottom", fontsize=9, color="0.35", style="italic")

    ymin, ymax = -np.pi - 0.2, np.pi + 0.7 + 0.55 * n
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-1.5, DSWE_MAX_CM)
    ax.set_xlabel(r"$\Delta$SWE (cm)")
    # center the y-label on the phase panel (bottom portion), not the full axis
    ax.set_ylabel(r"wrapped interferometric phase (rad)",
                  y=(0 - ymin) / (ymax - ymin))
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.15)

    out = FIG_DIR / "fig_phase_wraps_bands.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
