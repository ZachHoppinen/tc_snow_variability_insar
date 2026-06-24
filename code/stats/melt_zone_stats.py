"""melt_zone_stats.py -- |M| in the melt zones vs the accumulation zones of the
WY2023 pair (2 - 16 March 2023), the one melt-dominated pair.

WY2023 has both signs of dSWE: the lower-elevation half is losing snow (melt)
while the high alpine is still gaining (accumulation). Splitting the 81 m
windows by the sign of the window-mean dSWE gives a controlled within-pair,
within-basin comparison: same sensor, same dates, opposite sign. If melt removes
SWE more spatially uniformly than wind deposits it, the melt windows should hold
a higher |M| than accumulation windows of the SAME |dSWE| magnitude.

We report the two subsets, then a magnitude-matched comparison binned by
|window-mean dSWE| so the melt-vs-accumulation |M| gap is not just a magnitude
effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # repo code/ dir
from aso_M_stats import window_stats  # noqa: E402
from tc_paths import FIG_ROOT  # noqa: E402

FIG_DIR = FIG_ROOT / "stats"
PAIR = ("2023Mar02-03", "2023Mar16-17")     # WY2023 melt pair


def summarize(name, md, sd, am, ag):
    print(f"  {name:12s} n={md.size:8,d}  "
          f"med dSWE={np.median(md):+7.2f} cm  "
          f"med |dSWE|={np.median(np.abs(md)):6.2f} cm  "
          f"med sigma={np.median(sd):5.2f} cm  "
          f"med |M|={np.median(am):.3f}  "
          f"med arg={np.median(ag):+5.1f} deg")


def main():
    md, sd, sk, am, ag, ev = window_stats(*PAIR)  # cm, cm, -, |M|, deg, elev (L-band)
    melt = md < 0
    acc = md > 0

    print("WY2023 (2-16 March 2023), L-band, 81 m windows\n")
    summarize("melt (dSWE<0)", md[melt], sd[melt], am[melt], ag[melt])
    summarize("accum (dSWE>0)", md[acc], sd[acc], am[acc], ag[acc])
    summarize("all", md, sd, am, ag)

    # magnitude-matched: median |M| per |dSWE| bin, melt vs accumulation
    edges = np.array([5, 10, 15, 20, 25, 35])
    amag = np.abs(md)
    print("\n  magnitude-matched median |M| (melt vs accumulation):")
    print(f"  {'|dSWE| bin (cm)':>16s} {'n melt':>8s} {'|M| melt':>9s} "
          f"{'n acc':>8s} {'|M| acc':>8s}")
    cen, mlt, acc_m = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = melt & (amag >= lo) & (amag < hi)
        aa = acc & (amag >= lo) & (amag < hi)
        if mm.sum() < 50 or aa.sum() < 50:
            continue
        cen.append((lo + hi) / 2)
        mlt.append(np.median(am[mm]))
        acc_m.append(np.median(am[aa]))
        print(f"  {f'{lo}-{hi}':>16s} {int(mm.sum()):8,d} {mlt[-1]:9.3f} "
              f"{int(aa.sum()):8,d} {acc_m[-1]:8.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.plot(cen, mlt, "o-", color="tab:blue", label="melt (dSWE<0)")
    ax.plot(cen, acc_m, "o-", color="tab:red", label="accumulation (dSWE>0)")
    ax.set_xlabel("|window-mean dSWE| (cm)")
    ax.set_ylabel("median |M| (L-band, 81 m)")
    ax.set_title("WY2023: melt holds higher |M| than accumulation\n"
                 "at the same |dSWE| magnitude")
    ax.legend()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "melt_vs_accum_WY2023.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
