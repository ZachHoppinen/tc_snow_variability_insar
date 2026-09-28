"""Per-pair band comparison: |M| and arg(M) at C, L, P for one ASO pair (v4).

Idea: the SAME sub-pixel dSWE field decorrelates a multilooked pixel MORE at
shorter wavelength, because the SWE-to-phase sensitivity scales as 1/lambda,

    kappa(lambda, theta) = (2*pi/lambda) (1.59 + theta^2.5)   [rad / m SWE]
    kappa_C : kappa_L : kappa_P  ~  4.2 : 1 : 0.29            (theta = 40 deg)

so a fixed within-window spread in dSWE fans the per-cell phasors
exp(i*kappa*dSWE) much further at C than at P. The window factor

    M = < exp(i*kappa*dSWE) > * exp(-i*kappa*<dSWE>)

therefore has the SMALLEST |M| (most coherence loss) and the LARGEST |arg(M)|
(most phase bias) at C-band, and the mildest at P. This figure shows that band
ordering directly: a 2 x 3 grid, rows |M| (coherence factor in [0, 1]) and
arg(M) (phase bias, deg), columns C / L / P, for one ASO pair. M fields are
read from the per-band files written by aso_M_calculation.py.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plotting_constants import (  # noqa: E402
    apply_style, DPI, PROCESSED_DIR, FIG_DIR, CMAP_COH, CMAP_ARG, ARG_VLIM_DEG)

PAIR = ("WY2025", "2025Feb08-09", "2025Feb25")   # (label, early, late)
BANDS = ["C", "L", "P"]


def gap_days(a: str, b: str) -> int:
    def start(tok: str) -> datetime:
        m = re.match(r"(\d{4})([A-Za-z]{3})(\d{2})", tok)
        return datetime.strptime("".join(m.groups()), "%Y%b%d")
    return (start(b) - start(a)).days


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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    wy, a, b = PAIR
    pdir = PROCESSED_DIR / f"{a}_{b}"

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.7), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.03, w_pad=0.02,
                                    hspace=0.04, wspace=0.02)

    im_abs = im_arg = None
    for c, band in enumerate(BANDS):
        absM, argM, ext, org = load_M(pdir, band)
        im_abs = axes[0, c].imshow(absM, extent=ext, origin=org, cmap=CMAP_COH,
                                   vmin=0, vmax=1, interpolation="nearest")
        im_arg = axes[1, c].imshow(argM, extent=ext, origin=org, cmap=CMAP_ARG,
                                   vmin=-ARG_VLIM_DEG, vmax=ARG_VLIM_DEG,
                                   interpolation="nearest")
        axes[0, c].set_title(f"{band}-band")
        for r in range(2):
            axes[r, c].tick_params(labelleft=(c == 0), labelbottom=(r == 1))
            if r == 1:
                axes[r, c].set_xlabel("Easting (km)")

    axes[0, 0].set_ylabel("|M| - coherence factor\nNorthing (km)")
    axes[1, 0].set_ylabel("arg(M) - phase bias\nNorthing (km)")

    fig.colorbar(im_abs, ax=axes[0, :], location="right", shrink=0.85,
                 pad=0.02, label="Coherence")
    fig.colorbar(im_arg, ax=axes[1, :], location="right", shrink=0.85,
                 pad=0.02, label="Phase Bias (deg)")

    fig.suptitle(f"{wy} ({gap_days(a, b)} days): M by band at 81 m")
    out = FIG_DIR / f"7_bands_M_{wy}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
