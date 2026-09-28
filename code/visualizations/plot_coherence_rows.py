"""NISAR multi-resolution coherence vs ASO |M| (v4).

The hypothesis test for sub-pixel coherence LOSS. Sub-pixel dSWE spread shrinks
the resultant phasor, so a coarser (80 m) footprint should decorrelate more
than a finer (20 m) one -- by 1 - |M| in the ideal case. Finite looks also bias
coherence upward, more so at fewer looks, so we first de-bias each resolution by
inverting the Bamler curve at its own effective looks L,

    gamma_tilde = E^{-1}{ gamma_hat | L },     L_20 ~ 23, L_80 ~ 158,

and then take the corrected drop

    dgamma = gamma_tilde_20 - gamma_tilde_80   (NISAR sub-pixel coherence loss),

which column 2 compares against the ASO prediction |M| in column 3. P12 is the
low-snow pair (expect dgamma ~ 0); P23 is the large-storm pair.

Layout: one row per NISAR pair (P12, P23), three columns --
  1. gamma_80          observed 80 m coherence (raw)
  2. corrected y20-y80 Bamler-corrected 20 m minus 80 m coherence (dgamma)
  3. ASO |M|           WY2025 L-band coherence factor (same both rows)

All NISAR products (UTM 10N, 20/80 m) are reprojected onto the ASO 81 m grid
(UTM 11N) so every panel shares the grid; coherence is a magnitude, so block
averaging on reprojection is appropriate.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo code/ dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nisar"))  # debias helpers
from tc_paths import DATA_ROOT  # noqa: E402
from debias_coherences import debiaser, L_20  # noqa: E402  (shared Bamler inversion)

COH_DIR = DATA_ROOT / "sub_pixel_variability/processed/nisar/coherence"
from plotting_constants import (  # noqa: E402
    apply_style, DPI, PROCESSED_DIR, FIG_DIR, CMAP_COH, CMAP_COHDIFF, CMAP_NISAR_COH)

ASO_PAIR = ("2023Mar02-03", "2023Mar16-17")        # |M| panel source (WY2023, largest footprint)
NISAR_MASK_PAIR = ASO_PAIR                          # mask NISAR to the same WY2023 footprint
BAND = "L"
ROWS = [("P12", "December 7-19, 2025\n(no accumulation)"),
        ("P23", "December 19-31, 2025\n(large storm)")]


def aso_range(date_a: str, date_b: str) -> str:
    """Readable ASO acquisition range, e.g. 'February 8-25, 2025'."""
    def d(tok: str) -> datetime:
        m = re.match(r"(\d{4})([A-Za-z]{3})(\d{2})", tok)
        return datetime.strptime("".join(m.groups()), "%Y%b%d")
    da, db = d(date_a), d(date_b)
    if (da.year, da.month) == (db.year, db.month):
        return f"{da:%B} {da.day}-{db.day}, {da.year}"
    return f"{da:%B} {da.day} - {db:%B} {db.day}, {db.year}"


def load_absM(pair: tuple[str, str], band: str) -> xr.DataArray:
    """ASO |M| for a pair as a CRS-aware DataArray (all pairs share the grid)."""
    pdir = PROCESSED_DIR / f"{pair[0]}_{pair[1]}"
    with xr.open_dataset(pdir / f"aso_M_81m_{band}.nc") as ds:
        absM = np.abs(ds["M_real"].values + 1j * ds["M_imag"].values)
        da = xr.DataArray(absM.astype("float32"),
                          dims=ds["M_real"].dims, coords=ds["M_real"].coords)
    da.rio.write_crs("EPSG:32611", inplace=True)
    return da


def reproj(name: str, template: xr.DataArray) -> np.ndarray:
    """Reproject a coherence tif onto the template grid (block average)."""
    with rioxarray.open_rasterio(COH_DIR / name, masked=True) as da0:
        da = da0.squeeze(drop=True).load()
    return da.rio.reproject_match(template, resampling=Resampling.average).values


_DEB20 = debiaser(L_20, gmax=0.97)


def coarsen4(a: np.ndarray) -> np.ndarray:
    """NaN-safe 4x4 block mean: native 20 m grid -> aligned 80 m grid."""
    ny, nx = (a.shape[0] // 4) * 4, (a.shape[1] // 4) * 4
    return np.nanmean(a[:ny, :nx].reshape(ny // 4, 4, nx // 4, 4), axis=(1, 3))


def ad_diff(prefix: str, template: xr.DataArray) -> np.ndarray:
    """Corrected dgamma via average-then-debias (calibration-consistent): block-
    average the raw 20 m to the 80 m grid, de-bias at L_20, subtract the de-biased
    80 m, then reproject onto the display template. (De-biasing native 20 m pixels
    instead leaves a ~+0.03 bias over open water.)"""
    with rioxarray.open_rasterio(COH_DIR / f"{prefix}_coherence_raw_20m.tif",
                                 masked=True) as d:
        r20 = d.squeeze(drop=True).load().values
    with rioxarray.open_rasterio(COH_DIR / f"{prefix}_coherence_corrected_80m.tif",
                                 masked=True) as d:
        c80da = d.squeeze(drop=True).load()
    avg = coarsen4(r20)
    c20 = np.full_like(avg, np.nan)
    fin = np.isfinite(avg)
    c20[fin] = _DEB20(avg[fin])
    dgda = c80da.copy(data=(c20 - c80da.values))
    return dgda.rio.reproject_match(template, resampling=Resampling.average).values


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    template = load_absM(ASO_PAIR, BAND)         # |M| panel (WY2025)
    absM = template.values
    maskM = np.isfinite(absM)                    # |M| footprint (WY2025)
    maskN = np.isfinite(load_absM(NISAR_MASK_PAIR, BAND).values)  # NISAR footprint (WY2025, = |M| footprint)

    # gather per-row panels on the shared grid
    panels = []      # (row_label, gamma80, corrected_diff)
    for prefix, label in ROWS:
        g80 = reproj(f"{prefix}_coherence_raw_80m.tif", template)
        diff = ad_diff(prefix, template)         # average-then-debias dgamma
        panels.append((label, g80, diff))

    # NISAR coverage within the WY2025 mask
    nisar_cov = maskN.copy()
    for _, g80, _ in panels:
        nisar_cov = nisar_cov & np.isfinite(g80)
    # crop to the WY2025 footprint (NISAR and |M| share it)
    cov = nisar_cov | maskM
    rows_i = np.where(cov.any(axis=1))[0]
    cols_i = np.where(cov.any(axis=0))[0]
    sl = (slice(rows_i.min(), rows_i.max() + 1),
          slice(cols_i.min(), cols_i.max() + 1))
    x = template.x.values[sl[1]] / 1e3
    y = template.y.values[sl[0]] / 1e3
    ext = (x.min(), x.max(), y.min(), y.max())
    org = "upper" if template.y.values[0] > template.y.values[-1] else "lower"

    # symmetric scale for the difference column, robust across both rows
    dvmax = float(np.nanpercentile(np.abs(np.concatenate(
        [np.where(maskN, p[2], np.nan)[sl].ravel() for p in panels])), 98))

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.03, w_pad=0.02,
                                    hspace=0.03, wspace=0.03)
    gs = fig.add_gridspec(2, 3)
    # cols 0,1: one axis per row; col 2: a single panel nested in a 3-row
    # sub-grid so it sits in the vertical middle -> reads as one shared dataset.
    ax_nisar = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]
    sub = gs[:, 2].subgridspec(3, 1, height_ratios=[1, 2, 1])
    ax_M = fig.add_subplot(sub[1, 0])

    col_titles = [r"$\gamma_{80}$ (observed)",
                  r"corrected $\gamma_{20}-\gamma_{80}$"]
    kw_nisar = [dict(cmap=CMAP_NISAR_COH, vmin=0, vmax=1),
                dict(cmap=CMAP_COHDIFF, vmin=-dvmax, vmax=dvmax)]
    im_cols = [None, None]
    for r, (label, g80, diff) in enumerate(panels):
        arrs = [np.where(maskN, g80, np.nan)[sl],
                np.where(maskN, diff, np.nan)[sl]]
        for c in range(2):
            ax = ax_nisar[r][c]
            im_cols[c] = ax.imshow(arrs[c], extent=ext, origin=org,
                                   interpolation="nearest", **kw_nisar[c])
            ax.tick_params(labelleft=(c == 0), labelbottom=(r == 1))
            if r == 0:
                ax.set_title(col_titles[c])
            if r == 1:
                ax.set_xlabel("Easting (km)")
        ax_nisar[r][0].set_ylabel(f"{label}\nNorthing (km)")

    # single ASO |M| panel, centered in column 3
    im_M = ax_M.imshow(absM[sl], extent=ext, origin=org, cmap=CMAP_COH,
                       vmin=0, vmax=1, interpolation="nearest")
    ax_M.set_title(f"ASO |M|\n{aso_range(*ASO_PAIR)}")
    ax_M.set_xlabel("Easting (km)")
    ax_M.tick_params(labelleft=False)

    fig.colorbar(im_cols[0], ax=[ax_nisar[0][0], ax_nisar[1][0]],
                 location="bottom", shrink=0.85, pad=0.02, label="Coherence")
    fig.colorbar(im_cols[1], ax=[ax_nisar[0][1], ax_nisar[1][1]],
                 location="bottom", shrink=0.85, pad=0.02,
                 label=r"$\Delta$ Coherence")
    fig.colorbar(im_M, ax=ax_M, location="bottom", shrink=0.85, pad=0.02,
                 label="|M|")

    out = FIG_DIR / "6_coherence_rows_P12_P23.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
