"""plot_viirs_snowcover.py -- Appendix B figure: VIIRS fSCA beside the corrected
gamma_20 - gamma_80, two rows (P12 no-accumulation, P23 storm), matching the row
order of Figure~\\ref{fig:coherence_rows}.

All map building lives in stats/viirs_snowcover_stats.py (the source of truth for
the Appendix B numbers); this script imports build_maps()/ROWS and only plots.
Run viirs/fetch_viirs_snowcover.py first to populate raw/viirs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stats"))
from viirs_snowcover_stats import build_maps, ROWS  # noqa: E402

from plotting_constants import (  # noqa: E402
    apply_style, DPI, FIG_DIR, CMAP_COHDIFF)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    template, mask, maps = build_maps()

    # crop to the WY2023 footprint bounding box
    ri = np.where(mask.any(axis=1))[0]
    ci = np.where(mask.any(axis=0))[0]
    sl = (slice(ri.min(), ri.max() + 1), slice(ci.min(), ci.max() + 1))
    x = template.x.values[sl[1]] / 1e3
    y = template.y.values[sl[0]] / 1e3
    ext = (x.min(), x.max(), y.min(), y.max())
    org = "upper" if template.y.values[0] > template.y.values[-1] else "lower"

    # symmetric diff scale, robust across both rows
    all_dg = np.concatenate([maps[p[0]]["dgamma"][sl].ravel() for p in ROWS])
    dvmax = float(np.nanpercentile(np.abs(all_dg[np.isfinite(all_dg)]), 98))

    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), constrained_layout=True)
    im_snow = im_dg = None
    for r, (prefix, _, _, label) in enumerate(ROWS):
        im_snow = axes[r][0].imshow(maps[prefix]["fsca"][sl], extent=ext,
                                    origin=org, cmap="Blues", vmin=0, vmax=1,
                                    interpolation="nearest")
        im_dg = axes[r][1].imshow(maps[prefix]["dgamma"][sl], extent=ext,
                                  origin=org, cmap=CMAP_COHDIFF,
                                  vmin=-dvmax, vmax=dvmax, interpolation="nearest")
        axes[r][0].set_ylabel(f"{label}\nNorthing (km)")
        axes[r][1].tick_params(labelleft=False)
        for c in (0, 1):
            axes[r][c].tick_params(labelbottom=(r == len(ROWS) - 1))
            if r == len(ROWS) - 1:
                axes[r][c].set_xlabel("Easting (km)")
    axes[0][0].set_title("VIIRS fSCA")
    axes[0][1].set_title(r"corrected $\gamma_{20}-\gamma_{80}$")

    fig.colorbar(im_snow, ax=[axes[0][0], axes[1][0]], location="bottom",
                 shrink=0.85, pad=0.02, label="fSCA")
    fig.colorbar(im_dg, ax=[axes[0][1], axes[1][1]], location="bottom",
                 shrink=0.85, pad=0.02, label=r"$\Delta$ Coherence")

    out = FIG_DIR / "B1_snowcover_vs_dgamma.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
