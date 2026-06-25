"""plot_dswe_psd_break.py -- figure: per-window radial PSD of the native 3 m ASO
dSWE field with the two-regime fractal break (Figure~\\ref{fig:psd_break}).

All computation lives in stats/psd_break_stats.py (the source of truth for the
break numbers); this script imports compute_window_breaks() and only plots, so
the figure and the reported statistics stay in sync. See that script for method.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stats"))
from psd_break_stats import compute_window_breaks  # noqa: E402

from plotting_constants import (  # noqa: E402
    apply_style, DPI, FIG_DIR)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    R = compute_window_breaks()
    wl, use, specs, mean_lp = R["wl"], R["use"], R["specs"], R["mean_lp"]
    mlam, mbs, mbl, mcoef = R["mlam"], R["mbs"], R["mbl"], R["mcoef"]

    fig, axe = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for j in range(len(specs)):                    # every window, faint
        axe.loglog(wl[use], np.exp(specs[j][use]), color="0.45", lw=0.5, alpha=0.12)
    axe.loglog(wl[use], np.exp(mean_lp[use]), color="tab:blue", lw=4.5,
               label="mean spectrum", zorder=3)   # thick, sits under the lines

    # the two power-law segments as straight lines that cross at the break,
    # each spanning the full data range so they run edge to edge.
    xb = np.log(mlam)
    s_small, i_small = mcoef[1], mcoef[0]              # small-wavelength line
    s_large = mcoef[1] + mcoef[2]                      # large-wavelength line
    i_large = mcoef[0] - mcoef[2] * xb
    wl_min, wl_max = wl[use].min(), wl[use].max()      # grey-line data extent
    xx = np.array([np.log(wl_min), np.log(wl_max)])
    axe.loglog(np.exp(xx), np.exp(i_small + s_small * xx), color="tab:orange",
               lw=2.2, zorder=4, label=f"small-scale  $\\beta$={mbs:.1f}")
    axe.loglog(np.exp(xx), np.exp(i_large + s_large * xx), color="tab:green",
               lw=2.2, zorder=4, label=f"large-scale  $\\beta$={mbl:.1f}")
    ybreak = np.exp(i_small + s_small * xb)
    axe.axvline(mlam, color="0.5", ls=":", lw=1, zorder=2)
    axe.plot(mlam, ybreak, "o", color="k", ms=8, zorder=5,
             label=f"break = {mlam:.0f} m")

    axe.set_xlim(wl_max, wl_min)                       # no gap past the data
    axe.set_xlabel("wavelength (m)")
    axe.set_ylabel("normalized PSD")
    axe.legend(fontsize=9, loc="lower left")
    out = FIG_DIR / "3_dswe_psd_break.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
