"""window_M_arg_vs_size.py -- the two MEASURED sub-window effects vs multilook
window size, pooled over all four ASO pairs, straight from the 3 m dSWE.

Two stacked subplots, window sizes 9 to 81 m (native 3 m cells, common snow
pixels), pooled over the four pairs:

  top    |M|(W)     median + IQR  -- sub-window coherence loss
  bottom arg(M)(W)  median + IQR in mm SWE, over USABLE windows (|M| > USABLE_M)
                    -- systematic phase bias that survives a coherence threshold

arg(M) is conditioned on |M| > USABLE_M because decorrelated windows have arg(M)
random over +-pi (they would be masked from a retrieval anyway). Pooling is done
by accumulating per-size histograms across the four pairs, so percentiles come
from the full pooled window population without holding every window in memory.
L-band. Raw 3 m carries ASO depth noise, so |M| is a lower bound on the true
coherence (see window_preavg_compare.py).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "exploratory"))   # window_size_sweep helpers
from window_size_sweep import (  # noqa: E402
    M_for_side, build_or_load_common_mask, load_dswe_3m, PAIRS, KAPPA, FIG_DIR,
)

SIDES = list(range(3, 28))                   # 3..27 cells -> 9..81 m
SIZES_M = np.array([s * 3 for s in SIDES])
USABLE_M = 0.3                               # coherence cut for the bias
M_EDGES = np.linspace(0.0, 1.0, 1001)        # |M| histogram bins
A_EDGES = np.linspace(-np.pi, np.pi, 1801)   # arg(M) histogram bins


def pctiles_from_hist(counts, edges, qs):
    """Percentiles (list qs in %) from a histogram via CDF interpolation."""
    c = np.cumsum(counts)
    if c[-1] == 0:
        return [np.nan] * len(qs)
    cdf = c / c[-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return list(np.interp([q / 100.0 for q in qs], cdf, centers))


def main() -> None:
    common = build_or_load_common_mask()

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

    # pooled percentiles per size
    Mp = np.array([pctiles_from_hist(Mhist[si], M_EDGES, [25, 50, 75])
                   for si in range(nsz)])                       # (nsz, 3)
    Ap = np.array([pctiles_from_hist(Ahist[si], A_EDGES, [25, 50, 75])
                   for si in range(nsz)]) / KAPPA * 1000.0      # mm SWE

    print(f"\n{'W(m)':>5s} {'|M| p25':>8s} {'med':>6s} {'p75':>6s}   "
          f"{'arg p25':>8s} {'med':>6s} {'p75':>6s}  (mm)")
    for si, w in enumerate(SIZES_M):
        print(f"{w:5d} {Mp[si,0]:8.3f} {Mp[si,1]:6.3f} {Mp[si,2]:6.3f}   "
              f"{Ap[si,0]:8.2f} {Ap[si,1]:6.2f} {Ap[si,2]:6.2f}")

    fig, (axM, axA) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   constrained_layout=True)

    axM.fill_between(SIZES_M, Mp[:, 0], Mp[:, 2], color="tab:blue", alpha=0.2)
    axM.plot(SIZES_M, Mp[:, 1], "o-", color="tab:blue", lw=2)
    axM.set_ylabel("|M|  (coherence factor)")
    axM.set_ylim(0, 1.0)
    axM.set_title("Sub-window coherence loss and phase bias vs window size\n"
                  "(L-band, 3 m ASO dSWE, pooled over 4 pairs, median and IQR)")

    axA.fill_between(SIZES_M, Ap[:, 0], Ap[:, 2], color="tab:red", alpha=0.2)
    axA.plot(SIZES_M, Ap[:, 1], "s-", color="tab:red", lw=2)
    axA.axhline(0, color="gray", lw=0.8)
    axA.set_ylabel(f"arg(M) bias (mm SWE), |M|>{USABLE_M:g}")
    axA.set_xlabel("multilook window size (m)")

    for ax in (axM, axA):
        ax.axvline(80, color="gray", ls=":", lw=1)
    axM.text(80, 0.98, " 80 m GUNW", va="top", ha="right", fontsize=8, color="gray")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "window_M_arg_vs_size_L.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
