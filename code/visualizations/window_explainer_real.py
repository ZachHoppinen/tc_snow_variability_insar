"""Theory-section companion to window_explainer.py: the same multi-looking
shown on a slice of a real single-look NISAR interferogram.

Three panels of wrapped phase over the same patch in radar coordinates
(slant range x along-track, both in meters):
  (a) single look             every resolution cell, with the 5 x 6 block
                              grid that feeds panel (b) drawn on it
  (b) 5 x 6 looks             the GUNW wrapped-layer window (~20 m posting),
                              with the 13 x 16 window grid that feeds (c)
  (c) 13 x 16 looks           the GUNW unwrapped-layer window (~80 m posting)
One 5 x 6 and one 13 x 16 window are also outlined at the same ground size
in the corner of every panel.

Multi-looking is the box mean of the complex single-look interferogram with
decimation, the same operation the GUNW metadata reports ("Spatial moving
average with decimation"). Only the phase of the result is shown, after a
constant reference (the patch's circular-mean phase) is removed for display.

Pixel-exact layout: every single-look cell is drawn as exactly PX x PY pixels
so a non-integer resampling of the cell grid cannot print false bands.

Input: the single-look RIFG (wrappedInterferogram, complex64) from the
cropped Tuolumne run in code/nisar/onelook/.
Usage: window_explainer_real.py [AOI] [--both]
  AOI    block name from nisar/onelook/onelook_config.py (default cs5, the
         block used in the paper: 1 km, 3047 m mean elevation)
  --both add the no-accumulation P12 pair as a second row
  --small a 4 x 3 window piece with cells drawn large and one window followed
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

from plotting_constants import apply_style, FIG_DIR

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
AOI = ARGS[0] if ARGS else "cs5"
ONELOOK = Path(__file__).resolve().parents[2] / "data/processed/nisar/onelook" / AOI
PAIRS = [("P12", "20251207_20251219")] if "--p12" in sys.argv else [("P23", "20251219_20251231")]
if "--both" in sys.argv:
    PAIRS.append(("P12", "20251207_20251219"))
PAIRS = [(lab, d) for lab, d in PAIRS if (ONELOOK / d / "scratch/RIFG.h5").exists()]

GRP = "science/LSAR/RIFG/swaths/frequencyA/interferogram"
POL = "HH"
LOOKS = [(1, 1), (5, 6), (13, 16)]          # (range, azimuth) looks per panel
TITLES = ["Single look", "5 x 6 looks", "13 x 16 looks"]
# patch size in single-look cells: a common multiple of both windows so every
# panel covers exactly the same ground (260 = 4*65 range, 240 = 5*48 azimuth,
# about 1.2 x 1.1 km)
NR, NA = 260, 240
# --small: a piece of NWIN_R x NWIN_A 13 x 16 windows instead of the 1 km patch,
# with one window (FOLLOW, column/row from lower left) outlined on every panel
SMALL = "--small" in sys.argv
_n = [a for a in sys.argv[1:] if a.startswith("--nwin=")]
NWIN_R, NWIN_A = (tuple(int(v) for v in _n[0].split("=")[1].split(","))) if _n else (4, 3)
FOLLOW = (2, 1)
# --origin C,R  the 13 x 16 window (column, row from the lower left of the 1 km
#               patch) at the lower-left corner of the small piece
# --p12         use the no-accumulation Dec 7 -> 19 pair instead of the storm pair
_o = [a for a in sys.argv[1:] if a.startswith("--origin=")]
ORIGIN = tuple(int(v) for v in _o[0].split("=")[1].split(",")) if _o else (10 - FOLLOW[0], 12 - FOLLOW[1])
CMAP = "twilight_shifted"
BOX80, BOX20 = "gold", "tab:orange"
HALO = [pe.withStroke(linewidth=6, foreground="white")]   # keeps the boxes visible over any colour
GRID_KW = dict(color="k", lw=0.4, alpha=0.5)   # the window grid feeding the next panel


def multilook(z: np.ndarray, rl: int, al: int, partial: bool = False) -> np.ndarray:
    """Box mean of a complex field over rl x al blocks, with decimation. With
    partial=True the trailing edge blocks that do not fill a full rl x al box
    are kept as the mean of the cells they do contain."""
    na, nr = z.shape
    if not partial:
        z = z[: na - na % al, : nr - nr % rl]
        return z.reshape(na // al, al, nr // rl, rl).mean(axis=(1, 3))
    nb_a, nb_r = -(-na // al), -(-nr // rl)          # ceil
    out = np.empty((nb_a, nb_r), np.complex64)
    for j in range(nb_a):
        for i in range(nb_r):
            out[j, i] = z[j * al:(j + 1) * al, i * rl:(i + 1) * rl].mean()
    return out


def load_patch(rifg: Path):
    """Central NA x NR single-look patch of one RIFG, plus its cell spacings.
    In --small mode, the piece of NWIN_R x NWIN_A windows starting at the
    high-coherence window (10, 12) of the cs5 patch, offset so FOLLOW lands on it."""
    with h5py.File(rifg) as h:
        ifg = h[f"{GRP}/{POL}/wrappedInterferogram"]
        a0, r0 = (ifg.shape[0] - NA) // 2, (ifg.shape[1] - NR) // 2
        if SMALL:
            r0 += ORIGIN[0] * 13; a0 += ORIGIN[1] * 16
            sl = (slice(a0, a0 + NWIN_A * 16), slice(r0, r0 + NWIN_R * 13))
        else:
            sl = (slice(a0, a0 + NA), slice(r0, r0 + NR))
        z = ifg[sl].astype(np.complex64)
        dr = float(h[f"{GRP}/slantRangeSpacing"][()])               # m per cell, slant range
        da = float(h[f"{GRP}/sceneCenterAlongTrackSpacing"][()])    # m per cell
    # display only: rotate the patch so its circular-mean phase sits at zero,
    # keeping the bulk of the cells away from the +-pi wrap of the colormap
    ref = np.angle(np.exp(1j * np.angle(z)).mean())
    print(f"  reference phase removed: {ref:+.2f} rad")
    return z * np.exp(-1j * ref), dr, da


def draw_grid(ax, rl, al, dr, da):
    """Every rl x al window boundary across the patch."""
    for i in range(0, NR + 1, rl):
        ax.axvline(i * dr, **GRID_KW)
    for j in range(0, NA + 1, al):
        ax.axhline(j * da, **GRID_KW)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    nrow = len(PAIRS)
    # Pixel-exact layout: every single-look cell is drawn as exactly PX x PY
    # pixels (7 x 10 matches the 3.12 x 4.45 m cell aspect).
    # cell pixel size shrinks with the piece so the figure stays ~6300 px wide
    PX, PY, DPI_OUT = ((21, 30, 300) if SMALL else (7, 10, 300))   # 7:10 = cell aspect
    nr, na = (NWIN_R * 13, NWIN_A * 16) if SMALL else (NR, NA)
    pw, ph = nr * PX, na * PY                          # panel size in pixels
    left, gap, top, bottom, cbar_w = 260, 150, 120, 170, 360
    fw = left + 3 * pw + 2 * gap + cbar_w
    fh = top + nrow * ph + (nrow - 1) * gap + bottom
    fig = plt.figure(figsize=(fw / DPI_OUT, fh / DPI_OUT), dpi=DPI_OUT)
    axes = np.empty((nrow, 3), dtype=object)
    for row in range(nrow):
        y0 = fh - top - (row + 1) * ph - row * gap
        for col in range(3):
            x0 = left + col * (pw + gap)
            axes[row, col] = fig.add_axes([x0 / fw, y0 / fh, pw / fw, ph / fh])
    cb_y0 = (fh - top - nrow * ph - (nrow - 1) * gap) / fh
    cb_h = (nrow * ph + (nrow - 1) * gap) / fh
    cax = fig.add_axes([(left + 3 * pw + 2 * gap + 60) / fw, cb_y0, 40 / fw, cb_h])

    for row, (lab, d) in enumerate(PAIRS):
        z, dr, da = load_patch(ONELOOK / d / "scratch/RIFG.h5")
        print(f"{lab} {d}: patch {NA} x {NR} cells = {NA * da:.0f} m along-track x "
              f"{NR * dr:.0f} m slant range, {np.isfinite(z).mean() * 100:.1f}% finite")
        extent = (0, nr * dr, 0, na * da)          # same ground extent every panel
        for col, (ax, (rl, al), title) in enumerate(zip(axes[row], LOOKS, TITLES)):
            # in the small piece the 20 m blocks do not tile the piece exactly, so
            # the edge blocks are averaged over the cells they do contain; the image
            # extent still follows the full-block grid so blocks sit on the cell grid
            zm = multilook(z, rl, al, partial=SMALL)
            pext = (0, zm.shape[1] * rl * dr, 0, zm.shape[0] * al * da)
            vlim = 2.0 if SMALL else np.pi
            cmap = "RdBu_r" if SMALL else CMAP      # clipped range, so a plain diverging map
            im = ax.imshow(np.angle(zm), cmap=cmap, vmin=-vlim, vmax=vlim,
                           origin="lower", extent=pext, interpolation="nearest",
                           aspect="auto")
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            if SMALL:
                fx, fy = FOLLOW[0] * 13 * dr, FOLLOW[1] * 16 * da
                # pixel grid on the multi-looked panels only (the single-look cells
                # are too small for a grid to be anything but a mesh)
                if col > 0:
                    for i in range(0, nr + 1, rl):
                        ax.axvline(i * dr, color="0.4", lw=0.3, zorder=1)
                    for j in range(0, na + 1, al):
                        ax.axhline(j * da, color="0.4", lw=0.3, zorder=1)
                if col == 1:
                    # the 5 x 6 pixels the followed window overlaps: the 20 m layer is
                    # tiled from the piece origin, not the window corner, so these are
                    # snapped to that tiling and extend past the window edges
                    c0, c1 = FOLLOW[0] * 13, FOLLOW[0] * 13 + 13
                    a0_, a1_ = FOLLOW[1] * 16, FOLLOW[1] * 16 + 16
                    for i in range((c0 // 5) * 5, c1, 5):
                        for j in range((a0_ // 6) * 6, a1_, 6):
                            ax.add_patch(Rectangle((i * dr, j * da), 5 * dr, 6 * da, fill=False, ec=BOX20,
                                                   lw=2, zorder=3, path_effects=HALO))
                else:                   # the followed window on the single-look and 13 x 16 panels
                    ax.add_patch(Rectangle((fx, fy), 13 * dr, 16 * da, fill=False, ec=BOX80, lw=3.5,
                                           zorder=4, path_effects=HALO))
                title = ["Single look", "5 x 6 looks", "13 x 16 looks"][col]
            else:
                # the window grid that feeds the NEXT panel: 5 x 6 blocks on the
                # single-look panel, 13 x 16 windows on the 5 x 6 panel
                if col < 2:
                    draw_grid(ax, *LOOKS[col + 1], dr, da)
            if row == 0:
                ax.set_title(title)
            if row == nrow - 1:
                if SMALL:
                    from matplotlib.ticker import FuncFormatter, MultipleLocator
                    ax.xaxis.set_major_locator(MultipleLocator(26 * dr))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / dr:.0f}"))
                    ax.yaxis.set_major_locator(MultipleLocator(32 * da))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / da:.0f}"))
                    ax.set_xlabel("single-look range cells")
                else:
                    ax.set_xlabel("slant range (m)")
            print(f"  {title:14s} {zm.shape[0]} x {zm.shape[1]} cells, "
                  f"{rl * dr:.1f} x {al * da:.1f} m per cell")
        # the two multi-look windows drawn to scale, same corner, on every panel
        if not SMALL:
            x0, y0 = 1 * dr, NA * da - 17 * da             # one cell in from the corner
            for ax in axes[row]:
                for (rl, al), color in [((13, 16), BOX80), ((5, 6), BOX20)]:
                    ax.add_patch(Rectangle((x0, y0 + (16 - al) * da), rl * dr, al * da,
                                           fill=False, ec=color, lw=2.5))
        axes[row, 0].set_ylabel("single-look azimuth cells" if SMALL else (f"{lab}\nalong-track (m)" if nrow > 1 else "along-track (m)"))
        for ax in axes[row, 1:]:
            ax.set_yticklabels([])
    fig.colorbar(im, cax=cax, label="wrapped phase (rad)")
    tag = ("_both" if len(PAIRS) > 1 else "") + ("_p12" if "--p12" in sys.argv else "") + (f"_small_{ORIGIN[0]}_{ORIGIN[1]}" if SMALL else "")
    out = FIG_DIR / f"0b_window_explainer_real_{AOI}{tag}.png"
    fig.savefig(out, dpi=DPI_OUT)
    plt.close(fig)
    print(f"figure {fw} x {fh} px; single-look cell = {PX} x {PY} px")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
