"""Theory-section teaching figure: the three sub-pixel phase regimes.

Two rows, three columns. Top row is the sub-pixel dSWE field within one window
(a true window-mean plus per-cell variation); bottom row is the corresponding
phasor diagram, one grey phasor per cell at its absolute phase phi = kappa*dSWE.

  black arrow    = true mean phase phibar = kappa * mean(dSWE)  (unit, to circle)
  blue arrow     = resultant zbar = (1/N) sum exp(i phi) = exp(i phibar) * M
  firebrick line = 1-|M| = coherence drop (gap from |M| out to the unit circle)
  orange arc     = arg(M) = the phase bias (rotation from |M| point to zbar)

  (1) true stationarity        uniform field  -> zbar on the phibar ray, |M|~1
  (2) symmetric, varying dSWE  symmetric spread -> zbar still ON the ray, |M|<1
  (3) skewed dSWE              drift tail -> zbar TILTS off the ray (arg(M)!=0)

(1)->(2) isolates variance->coherence; (2)->(3) isolates skew->bias at matched
|M|. Deterministic (noise-free) -- the noise/Bamler story lives in the text.

Purely synthetic -- no data inputs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.ndimage import gaussian_filter


from plotting_constants import (  # noqa: E402
    apply_style, DPI, FIG_DIR, CMAP_TEACH)

KAPPA_L = 52.7          # rad/m, L-band SWE phase sensitivity at 40 deg incidence
N = 12                  # sub-pixel cells per side within the window
PHI_BAR = 0.7           # rad, illustrative true mean phase (~40 deg)
MEAN_DSWE = PHI_BAR / KAPPA_L * 100.0   # cm, true window-mean dSWE (~1.3 cm)


def _arrange(delta, template):
    """Place the delta VALUES into cells ranked by a smooth template, so the
    field looks spatially coherent while its distribution is exactly `delta`."""
    out = np.empty(delta.size)
    out[np.argsort(template.ravel())] = np.sort(delta)
    return out.reshape(template.shape)


def make_fields():
    """Three sub-pixel dSWE fields (cm) = a true window-mean (MEAN_DSWE) plus
    per-cell variation with EXACT phase statistics: uniform (|M|~1), symmetric
    (arg=0), skewed drift (arg<0). Each cell's absolute phase is kappa*dSWE."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:N, 0:N]

    # phase deviations (rad) for each regime
    d1 = 0.05 * rng.standard_normal(N * N)               # (1) ~uniform -> |M|~1
    A = 1.6                                              # (2) symmetric, max|d|<pi
    half = np.linspace(0, A, N * N // 2)                 #     |M| = sinc(A) ~ 0.62
    d2 = np.concatenate([half, -half])                   #     mirror-symmetric -> arg=0
    base = rng.uniform(-0.9, 0.9, N * N)                 # (3) skewed: base + a sparse,
    boost = np.zeros(N * N)                              #     strong one-sided drift
    boost[: int(0.16 * N * N)] = 2.8                     #     -> arg ~ -20 deg, |d|<pi
    d3 = base + boost
    d3 -= d3.mean()

    # arrange each into a spatially-coherent pattern, convert rad -> cm deviation
    templates = [gaussian_filter(rng.standard_normal((N, N)), 1.5),     # mottled
                 xx.astype(float),                                      # gradient
                 np.exp(-(((xx - N * 0.70) ** 2 + (yy - N * 0.32) ** 2)  # drift blob
                          / (2 * (N * 0.17) ** 2)))]
    fields = [_arrange(d, t) / KAPPA_L * 100.0
              for d, t in zip((d1, d2, d3), templates)]
    # zero-mean the variation, then add the true window-mean dSWE
    return [MEAN_DSWE + (f - f.mean()) for f in fields]   # cm (true dSWE)


def draw_phasors(ax, field, cmap, norm):
    """Phasor diagram for a dSWE field: per-cell phasors coloured by their dSWE
    (same colormap as the field), black true-mean-phase arrow (phibar), blue
    resultant (zbar), red deviation arrow."""
    vals = field.ravel()
    phi = KAPPA_L * vals / 100.0                   # absolute per-cell phase (rad)
    zbar = np.exp(1j * phi).mean()                 # resultant
    phibar = KAPPA_L * field.mean() / 100.0        # true mean phase
    M = zbar * np.exp(-1j * phibar)                # bias factor: |M|, arg(M)

    ax.add_patch(Circle((0, 0), 1, fill=False, color="0.85", lw=1))
    # subtle real / imaginary axes
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.text(1.17, 0.02, "Re", color="0.6", fontsize=9, va="bottom", ha="right")
    ax.text(0.03, 1.17, "Im", color="0.6", fontsize=9, va="top", ha="left")
    # per-cell phasors, coloured by dSWE to tie each to its field cell
    for p, v in zip(phi, vals):
        ax.add_patch(FancyArrowPatch((0, 0), (np.cos(p), np.sin(p)),
                     arrowstyle="-", mutation_scale=5, color=cmap(norm(v)),
                     lw=1.0, alpha=0.9))
    # black: true mean phase phibar (unit arrow to the circle)
    ax.add_patch(FancyArrowPatch((0, 0), (np.cos(phibar), np.sin(phibar)),
                 shrinkA=0, shrinkB=0, arrowstyle="-|>", mutation_scale=15,
                 color="k", lw=2, zorder=4))
    # firebrick: the coherence DROP 1-|M| -- radial gap from the resultant
    # magnitude (|M|, along the true-phase direction) out to the unit circle.
    absM, argM = np.abs(M), np.angle(M)
    gt = (absM * np.cos(phibar), absM * np.sin(phibar))
    bt = (np.cos(phibar), np.sin(phibar))
    ax.plot([gt[0], bt[0]], [gt[1], bt[1]], color="firebrick", lw=4,
            zorder=6, solid_capstyle="butt")
    # blue: resultant zbar (length |M|, angle phibar + arg(M))
    ax.add_patch(FancyArrowPatch((0, 0), (zbar.real, zbar.imag),
                 shrinkA=0, shrinkB=0, arrowstyle="-|>", mutation_scale=20,
                 color="C0", lw=3, zorder=5))
    # orange: arg(M) -- the phase bias, a curved arrow at radius |M| rotating
    # from the green tip (no-bias landing) to the actual resultant zbar.
    if abs(argM) > np.deg2rad(1):
        a = np.linspace(phibar, phibar + argM, 40)
        ax.plot(absM * np.cos(a), absM * np.sin(a), color="tab:orange",
                lw=2.5, zorder=9, solid_capstyle="round")
        ax.add_patch(FancyArrowPatch((absM * np.cos(a[-2]), absM * np.sin(a[-2])),
                     (zbar.real, zbar.imag), shrinkA=0, shrinkB=0,
                     arrowstyle="-|>", mutation_scale=13, color="tab:orange", zorder=9))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.05, 1.2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return np.abs(M), np.degrees(np.angle(M))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    fields = make_fields()
    titles = ["Uniform $\\Delta$SWE", "Symmetric, varying $\\Delta$SWE",
              "Skewed $\\Delta$SWE"]
    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)

    # cells taller than wide -> both the square heatmaps (row 0) and the
    # equal-aspect phasor plots (row 1) become width-limited, so both fill the
    # full column width and align between rows.
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), constrained_layout=True)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP_TEACH)
    im = None
    for col, (field, title) in enumerate(zip(fields, titles)):
        im = axes[0, col].imshow(field, cmap=CMAP_TEACH, vmin=vmin, vmax=vmax)
        axes[0, col].set_title(title, fontsize=17)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        M, argd = draw_phasors(axes[1, col], field, cmap, norm)
        print(f"  {title:30s} |M|={M:.2f}, arg={argd:+.1f} deg")
    # legend for the arrows, on the upper-left panel
    axes[0, 0].legend(handles=[
        Line2D([0], [0], color="k", lw=2,
               label=r"$e^{i\bar\phi}$: true $\Delta$SWE phasor"),
        Line2D([0], [0], color="C0", lw=3, label=r"$\bar z = M\,e^{i\bar\phi}$"),
        Line2D([0], [0], color="firebrick", lw=4,
               label=r"$1-|M|$: coherence drop"),
        Line2D([0], [0], color="tab:orange", lw=2.5,
               label=r"$\arg(M)$: phase bias")],
        loc="lower left", fontsize=11, frameon=True)
    # attach to the whole grid (not just row 0) so both rows shrink equally and
    # stay column-aligned; the bar spans the full height on the right.
    fig.colorbar(im, ax=axes, shrink=0.6, label="$\\Delta$SWE (cm)")
    out = FIG_DIR / "teach_phasor_contrast.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
