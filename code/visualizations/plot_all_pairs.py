"""Overview figure: dSWE, |M|, and arg(M) for all four ASO pairs (v4).

Idea: survey how sub-pixel snow heterogeneity turns into an InSAR penalty
across the four Tuolumne accumulation events, which span a range of magnitude
and spatial roughness. For each pair the per-cell SWE phase is phi = kappa*dSWE
and the 81 m window factor is

    M = < exp(i*kappa*dSWE) > * exp(-i*kappa*<dSWE>),

so |M| (in [0, 1]) is the coherence retained and arg(M) the phase bias left by
the within-window dSWE distribution. Reading the columns left-to-right traces
the causal chain -- the dSWE field, the coherence drop |M| it causes, the phase
bias arg(M) it causes; reading top-to-bottom shows how that chain sharpens as
the event (and its sub-pixel variance) grows.

Layout: a 4 x 3 grid, one row per pair, columns = 81 m dSWE (block-averaged
from the 3 m map), |M|, arg(M). M is read per-pair/per-band from
aso_M_calculation.py (band selectable, default L = NISAR). All panels already
live on the 81 m window grid, so they register pixel-for-pixel.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from plotting_constants import (  # noqa: E402
    apply_style, DPI, PROCESSED_DIR, FIG_DIR, PAIRS, ARG_VLIM_DEG, CMAP_COH, CMAP_ARG, CMAP_DSWE)

BAND = "L"                    # which M field to show (C, L, or P)
PREAVG_3M_TO_81M = 27         # 3 m -> 81 m block factor (matches M window)


def gap_days(date_a: str, date_b: str) -> int:
    """Days between the two flight-start dates parsed from the tokens
    (e.g. '2023Mar02-03' -> 2023-03-02)."""
    def start(tok: str) -> datetime:
        m = re.match(r"(\d{4})([A-Za-z]{3})(\d{2})", tok)
        return datetime.strptime("".join(m.groups()), "%Y%b%d")
    return (start(date_b) - start(date_a)).days


def load_dswe_81m(pdir: Path) -> xr.DataArray:
    """3 m dSWE block-averaged to the 81 m window grid (cm)."""
    with xr.open_dataset(pdir / "aso_dswe_3m.nc") as ds:
        d = ds["dswe"].load()
    d81 = d.coarsen(x=PREAVG_3M_TO_81M, y=PREAVG_3M_TO_81M,
                    boundary="trim").mean(skipna=True)
    return d81 * 100.0        # m -> cm


def load_M(pdir: Path, band: str):
    """(|M|, arg(M) deg, extent_km, origin) from the per-band M file."""
    with xr.open_dataset(pdir / f"aso_M_81m_{band}.nc") as ds:
        M = ds["M_real"].values + 1j * ds["M_imag"].values
        x, y = ds["x"].values, ds["y"].values
    absM = np.abs(M)
    argM = np.degrees(np.angle(M))
    argM[~np.isfinite(absM)] = np.nan
    ext = (x.min() / 1e3, x.max() / 1e3, y.min() / 1e3, y.max() / 1e3)
    return absM, argM, ext, ("upper" if y[0] > y[-1] else "lower")


def extent_origin(da: xr.DataArray):
    x, y = da.x.values, da.y.values
    return ((x.min() / 1e3, x.max() / 1e3, y.min() / 1e3, y.max() / 1e3),
            "upper" if y[0] > y[-1] else "lower")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    # gather all panels first so dSWE can share one symmetric color scale
    data = []
    for wy, a, b in PAIRS:
        pdir = PROCESSED_DIR / f"{a}_{b}"
        dswe = load_dswe_81m(pdir)
        absM, argM, ext_m, org_m = load_M(pdir, BAND)
        data.append((wy, dswe, absM, argM, ext_m, org_m, gap_days(a, b)))
    dswe_vmax = float(np.nanpercentile(
        np.abs(np.concatenate([d[1].values.ravel() for d in data])), 98))

    fig, axes = plt.subplots(len(PAIRS), 3, figsize=(13, 12.5),
                             constrained_layout=True)
    # the map panels are wider than tall (equal aspect), so trim the inter-row
    # gap that would otherwise open up between rows.
    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.04,
                                    hspace=0.01, wspace=0.04)
    col_titles = [r"$\Delta$SWE (cm)", "|M| - coherence factor",
                  "arg(M) - phase bias"]
    cbar_labels = [r"$\Delta$SWE (cm)", "Coherence ()", "Phase Bias (deg)"]
    im_by_col = [None, None, None]
    for r, (wy, dswe, absM, argM, ext_m, org_m, gap) in enumerate(data):
        ext_d, org_d = extent_origin(dswe)

        im0 = axes[r, 0].imshow(dswe.values, extent=ext_d, origin=org_d,
                                cmap=CMAP_DSWE, vmin=-dswe_vmax, vmax=dswe_vmax,
                                interpolation="nearest")
        im1 = axes[r, 1].imshow(absM, extent=ext_m, origin=org_m,
                                cmap=CMAP_COH, vmin=0, vmax=1,
                                interpolation="nearest")
        im2 = axes[r, 2].imshow(argM, extent=ext_m, origin=org_m, cmap=CMAP_ARG,
                                vmin=-ARG_VLIM_DEG, vmax=ARG_VLIM_DEG,
                                interpolation="nearest")
        im_by_col = [im0, im1, im2]

        axes[r, 0].set_ylabel(f"{wy} ({gap} days)\nNorthing (km)")
        for c in range(3):
            # tick labels only on the outer frame: y on col 0, x on bottom row
            axes[r, c].tick_params(labelleft=(c == 0),
                                   labelbottom=(r == len(PAIRS) - 1))
            if r == 0:
                axes[r, c].set_title(col_titles[c])
            if r == len(PAIRS) - 1:
                axes[r, c].set_xlabel("Easting (km)")

    # one colorbar per column, spanning all rows
    for c, label in enumerate(cbar_labels):
        fig.colorbar(im_by_col[c], ax=axes[:, c], location="bottom",
                     shrink=0.85, pad=0.02, label=label)

    fig.suptitle(f"ASO dSWE, |M|, and arg(M) at 81 m  ({BAND}-band)",
                 fontsize=15)
    out = FIG_DIR / f"all_pairs_dswe_M_{BAND}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
