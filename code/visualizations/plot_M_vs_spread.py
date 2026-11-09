"""plot_M_vs_spread.py -- |M| and arg(M) as a function of the within-window
phase spread, from the moments of the within-window dSWE distribution, over
the real lidar windows (referee 2: quantify M against the within-window
standard deviation).

Per-cell phase deviation delta = kappa * (dSWE - window mean), with standard
deviation sigma (rad), skew S and excess kurtosis K. M = E[exp(i delta)] is the
characteristic function of delta, so log M is the cumulant series

  log M = -sigma^2/2 + K sigma^4/24 - ...  +  i(-S sigma^3/6 + ...)

  |M|    ~ exp(-sigma^2/2)                  Gaussian term (Rott 2003; Yueh 2026), exact for a Gaussian
  arg(M) ~ -S sigma^3/6                     skew sets the bias

Even moments set the magnitude, odd moments set the phase. The series holds
while the phase has not wrapped; on the lidar windows the |M| error is below
0.05 and the arg error below 3 deg up to about 1.5 rad, and both fail beyond
1.75 rad (stats/cumulant_check.py); the curves are drawn across the full axis.

The curves are drawn over a 2-D histogram of all 81 m lidar windows (four ASO
pairs, L-band kappa), from stats/aso_M_stats.window_stats. The arg(M) panel
shows only windows with |M| > USABLE_M, the paper's usable-coherence cut,
because the angle of a near-zero resultant is set by phase noise.

The per-window statistics are slow to build (the four 3 m fields are read), so
they are cached in CACHE; delete it, or pass --refresh, after changing the
pipeline.

Run from the repo root in the nisar_pytools env:
    python code/visualizations/plot_M_vs_spread.py [--refresh]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stats"))
from aso_M_stats import window_stats, PAIRS, KAPPA, PROCESSED_DIR  # noqa: E402
from plotting_constants import apply_style, DPI, FIG_DIR, COLOR_M, COLOR_ARG  # noqa: E402

SIGMA_MAX = 1.5 * np.pi                    # x range (rad); 15% of windows lie beyond it
SIGMA_FULL = np.linspace(0.0, SIGMA_MAX, 400)
SKEWS = [1.0]                              # skew curves for arg(M), drawn for both signs
USABLE_M = 0.3                             # arg(M) shown only where |M| passes the paper's usable cut
CACHE = PROCESSED_DIR / "cache_M_vs_spread.npz"   # pooled window stats


def absM_series(sigma, kurt=0.0):
    """|M| from the cumulant series to fourth order."""
    return np.exp(-sigma ** 2 / 2 + kurt * sigma ** 4 / 24)


def arg_series_deg(sigma, skew):
    """arg(M) from the cumulant series to third order, in degrees."""
    return np.degrees(-skew * sigma ** 3 / 6)


def lidar_windows():
    """Pooled per-window (sigma_rad, |M|, arg_deg, skew) over the four pairs."""
    sig, absM, arg, skew = [], [], [], []
    for wy, a, b in PAIRS:
        _, sd_cm, sk, am, ag, _ = window_stats(a, b)
        sig.append(KAPPA * sd_cm / 100.0)
        absM.append(am); arg.append(ag); skew.append(sk)
        print(f"  {wy}: {am.size:,} windows")
    return (np.concatenate(sig), np.concatenate(absM),
            np.concatenate(arg), np.concatenate(skew))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    if CACHE.exists() and "--refresh" not in sys.argv:
        print(f"loading cache {CACHE}")
        c = np.load(CACHE)
        sig, absM, arg_deg, skew = c["sig"], c["absM"], c["arg_deg"], c["skew"]
    else:
        print("loading lidar windows")
        sig, absM, arg_deg, skew = lidar_windows()
        np.savez(CACHE, sig=sig, absM=absM, arg_deg=arg_deg, skew=skew)
        print(f"wrote cache {CACHE}")

    fig, (ax_m, ax_a) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    # real windows as a 2-D histogram behind the curves
    hb_kw = dict(gridsize=60, cmap="Greys", norm=LogNorm(), mincnt=1, linewidths=0)
    ax_m.hexbin(sig, absM, extent=(0, SIGMA_MAX, 0, 1), **hb_kw)
    usable = absM > USABLE_M                 # arg is noise where the resultant is short
    hb = ax_a.hexbin(sig[usable], arg_deg[usable], extent=(0, SIGMA_MAX, -180, 180), **hb_kw)
    fig.colorbar(hb, ax=ax_a, pad=0.02, label="windows per bin")

    # cumulant-series curves: solid where the series holds, faded beyond
    def curve(ax, f, label, **st):
        ax.plot(SIGMA_FULL, f(SIGMA_FULL), label=label, **st)
    # Gaussian term is exact for that shape, so it is drawn solid across the axis
    ax_m.plot(SIGMA_FULL, absM_series(SIGMA_FULL), color=COLOR_M, lw=2, label=r"Gaussian, $e^{-\sigma_\delta^2/2}$")
    curve(ax_a, lambda x: arg_series_deg(x, 0.0), "symmetric", color=COLOR_ARG, lw=2)
    skew_color = {1.0: "tab:orange"}                              # solid lines, one color per S
    for sk in SKEWS:
        curve(ax_a, lambda x, sk=sk: arg_series_deg(x, sk), f"$S$ = $\\pm${sk:g}",
              color=skew_color[sk], lw=2)
        curve(ax_a, lambda x, sk=sk: arg_series_deg(x, -sk), None,
              color=skew_color[sk], lw=2)

    ax_m.set_ylabel("|M|  (coherence factor)")
    ax_a.set_ylabel("arg(M)  (deg)")
    ax_m.set_ylim(0, 1); ax_a.set_ylim(-180, 180)
    for ax in (ax_m, ax_a):
        ax.set_xlim(0, SIGMA_MAX)
        ax.set_xlabel(r"within-window phase spread $\sigma_\delta$ (rad)")
        # same spread as dSWE at L-band on the top axis
        top = ax.secondary_xaxis("top", functions=(lambda s: s / KAPPA * 100,
                                                    lambda c: c * KAPPA / 100))
        top.set_xlabel(r"within-window $\Delta$SWE spread, L-band (cm)")
    ax_m.legend(loc="upper right", frameon=False)
    ax_a.legend(loc="lower left", frameon=False)

    out = FIG_DIR / "1b_M_vs_spread.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote: {out}")

    # where the real windows sit, for the caption and text
    p = lambda x, q: np.percentile(x, q)
    print("\nlidar 81 m windows, pooled over four pairs (L-band)")
    print(f"  n = {sig.size:,}; usable (|M| > {USABLE_M}): {usable.sum():,}; "
          f"beyond the plotted {SIGMA_MAX:.2f} rad: {np.mean(sig > SIGMA_MAX) * 100:.1f}%")
    print(f"  sigma (rad): median {np.median(sig):.2f}, IQR {p(sig,25):.2f}-{p(sig,75):.2f}; "
          f"fraction below 1.5 rad: {np.mean(sig < 1.5):.2f}")
    print(f"  skew:        median {np.median(skew):+.2f}, IQR {p(skew,25):+.2f}-{p(skew,75):+.2f}; "
          f"fraction with |skew| < 0.5: {np.mean(np.abs(skew) < 0.5):.2f}")
    for s in (0.5, 1.0, 1.5):
        print(f"  sigma = {s:.1f} rad ({s / KAPPA * 100:.1f} cm): Gaussian |M| = {absM_series(s):.2f}, "
              f"bias at S = 1: {arg_series_deg(s, 1.0):+.0f} deg")


if __name__ == "__main__":
    main()
