"""3 m dSWE maps -> complex sub-pixel coherence factor M per 81 m window (v4).

Reads the per-pair 3 m dSWE maps from aso_pairwise_dswe.py, first 3 x 3
block-averages them to 9 m to beat down ASO's per-cell laser-shot noise
(random per-pixel scatter otherwise reads as sub-pixel decorrelation and
deflates |M|), then for every non-overlapping 81 m window (9 x 9 cells of the
9 m grid) computes the complex factor M that captures what sub-pixel dSWE
heterogeneity does to the multilooked InSAR observation:

    phi(cell) = kappa * dSWE(cell)               per-cell SWE phase  [rad]
    zbar      = < exp(i * phi) >                  window-mean phasor
    M         = zbar * exp(-i * kappa * <dSWE>)   zbar with the TRUE mean
                                                  window phase removed

So M is the ratio of the observed window phasor to the phasor a perfectly
homogeneous window would produce:
    |M|     = coherence factor in [0, 1]        (1 = no sub-pixel loss)
    arg(M)  = phase bias [rad]                   (0 = no bias; != 0 only if
              the sub-pixel dSWE distribution is skewed)

kappa is the Leinss et al. (2015) phase-per-SWE factor at NISAR's 40 deg
nominal incidence, evaluated for each of three bands -- C (Sentinel-1), L
(NISAR), and P (a long-wavelength bookend). The same pre-averaged 9 m dSWE
field feeds all three; only kappa changes. Pure ASO geometry -- no GUNW, no
reprojection.

Output per pair per band: a complex M field at 81 m posting (saved as
real/imag in aso_M_81m_<band>.nc) plus a two-panel |M| / arg(M) figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

# --- paths ------------------------------------------------------------------
# Per-pair folders written by aso_pairwise_dswe.py: each holds aso_dswe_3m.nc.
PROCESSED_DIR = Path("/Users/zmhoppinen/Documents/nisar_swe/data/"
                     "sub_pixel_variability/processed/aso")
FIG_DIR = Path("/Users/zmhoppinen/Documents/nisar_swe/figures/"
               "ASO_sweramp_unwrapping_coherence/v4")

# (water-year label, early flight, late flight) -- same pairs as upstream.
PAIRS = [
    ("WY2023", "2023Mar02-03",    "2023Mar16-17"),
    ("WY2024", "2024Jan29",       "2024Feb27-28"),
    ("WY2025", "2025Feb08-09",    "2025Feb25"),
    ("WY2026", "2026Jan31-Feb01", "2026Feb27-28"),
]

# --- geometry / physics -----------------------------------------------------
BASE_RES_M = 3                       # ASO 3 m dSWE posting
PREAVG = 3                           # 3 x 3 block-average 3 m -> 9 m (noise)
AVG_RES_M = BASE_RES_M * PREAVG      # 9 m working posting
WINDOW_M = 81                        # multilook footprint (~ NISAR 80 m)
BLOCK = WINDOW_M // AVG_RES_M        # 9 cells of 9 m per 81 m window
MIN_FRAC = 0.5             # need >= half a window's cells valid to keep it

# Leinss et al. (2015) phase-per-SWE factor:  kappa = (2*pi/lambda)(1.59 + theta^2.5)
# Evaluated at NISAR's nominal 40 deg incidence for each band's carrier.
INC_RAD = np.deg2rad(40.0)
BANDS = {"C": 0.056,    # Sentinel-1 C-band carrier (m)
         "L": 0.238,    # NISAR L-band carrier (1.26 GHz) (m)
         "P": 0.81}     # P-band wavelength bookend (no operational mission)


def kappa_of(wavelength_m: float) -> float:
    """Leinss phase-per-SWE factor [rad / m SWE] at the 40 deg incidence."""
    return (2 * np.pi / wavelength_m) * (1.59 + INC_RAD ** 2.5)


# Memory bound: form the complex field at most this many 9 m rows at a time.
CHUNK_ROWS = 4000


def pair_dir(date_a: str, date_b: str) -> Path:
    """The per-pair folder holding this pair's dSWE map and M output."""
    return PROCESSED_DIR / f"{date_a}_{date_b}"


def load_dswe(date_a: str, date_b: str) -> xr.DataArray:
    """Load one pair's 3 m dSWE map (carrying its CRS) from NetCDF.

    Loads into memory and closes the file (context manager) so no handle
    survives to interpreter shutdown.
    """
    with xr.open_dataset(pair_dir(date_a, date_b) / "aso_dswe_3m.nc") as ds:
        return ds["dswe"].load()


def preaverage(dswe: xr.DataArray, factor: int = PREAVG) -> xr.DataArray:
    """NaN-safe `factor` x `factor` block-mean of the 3 m dSWE -> 9 m.

    Averaging beats the per-cell ASO depth noise down by ~sqrt(factor^2). A
    9 m cell is valid if any of its 3 m cells were (coarsen skipna mean);
    edge growth is negligible at 81 m windowing.
    """
    out = dswe.coarsen(x=factor, y=factor, boundary="trim").mean(skipna=True)
    if dswe.rio.crs is not None:
        out.rio.write_crs(dswe.rio.crs, inplace=True)
    return out


def window_M(dswe: xr.DataArray, kappa: float,
             block: int = BLOCK, min_frac: float = MIN_FRAC) -> xr.DataArray:
    """Complex M per non-overlapping `block` x `block` window.

    Chunked over rows so the complex exp() never materialises the whole
    basin at once. Windows with fewer than `min_frac` valid cells come out
    NaN. Coordinates are the window-center UTM positions.
    """
    d_full = dswe.values
    ny, nx = d_full.shape
    nby, nbx = ny // block, nx // block          # count of whole windows
    nx2 = nbx * block                            # trim partial trailing cols
    M = np.full((nby, nbx), np.nan, dtype=np.complex64)

    rows_per_chunk = max(1, CHUNK_ROWS // block)
    for r0 in range(0, nby, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, nby)
        d = d_full[r0 * block:r1 * block, :nx2]
        valid = np.isfinite(d)

        # per-cell phasor (0 where invalid so it drops out of the sums)
        z = np.where(valid, np.exp(1j * kappa * np.nan_to_num(d)), 0.0)
        dz = np.where(valid, d, 0.0)

        # collapse each block to one value: sum over the two within-block axes
        shp = (r1 - r0, block, nbx, block)
        cnt = valid.reshape(shp).sum(axis=(1, 3))
        zbar = z.reshape(shp).sum(axis=(1, 3)) / np.maximum(cnt, 1)
        mean_d = dz.reshape(shp).sum(axis=(1, 3)) / np.maximum(cnt, 1)

        # remove the true window-mean phase -> M (bias + coherence drop)
        Mblk = zbar * np.exp(-1j * kappa * mean_d)
        ok = cnt >= min_frac * block * block
        M[r0:r1] = np.where(ok, Mblk, np.nan)

    # window-center coordinates from the (9 m) working-grid coords
    x = dswe.x.values[:nx2].reshape(nbx, block).mean(1)
    y = dswe.y.values[:nby * block].reshape(nby, block).mean(1)
    out = xr.DataArray(M, dims=("y", "x"), coords={"y": y, "x": x}, name="M")
    if dswe.rio.crs is not None:
        out.rio.write_crs(dswe.rio.crs, inplace=True)
    out.attrs.update(kappa_rad_per_m=float(kappa), window_m=WINDOW_M,
                     base_res_m=BASE_RES_M, preavg_res_m=AVG_RES_M)
    return out


def save_M(M: xr.DataArray, pdir: Path, band: str) -> Path:
    """Persist complex M as real/imag NetCDF (complex isn't native to NC)."""
    ds = xr.Dataset({"M_real": M.real, "M_imag": M.imag})
    ds.attrs.update(M.attrs)
    out = pdir / f"aso_M_81m_{band}.nc"
    ds.to_netcdf(out)
    ds.close()
    return out


def plot_M(M: xr.DataArray, wy: str, band: str, kappa: float) -> Path:
    """Two-panel diagnostic: |M| (coherence drop) and arg(M) (phase bias)."""
    absM = np.abs(M.values)
    argM = np.degrees(np.angle(M.values))
    argM[~np.isfinite(absM)] = np.nan

    x, y = M.x.values, M.y.values
    ext = (x.min() / 1e3, x.max() / 1e3, y.min() / 1e3, y.max() / 1e3)
    org = "upper" if y[0] > y[-1] else "lower"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   constrained_layout=True)
    im0 = ax0.imshow(absM, extent=ext, origin=org, cmap="viridis",
                     vmin=0, vmax=1)
    fig.colorbar(im0, ax=ax0, shrink=0.8, label="|M|  (coherence factor)")
    ax0.set_title(f"|M|   (median {np.nanmedian(absM):.3f})")
    ax0.set_xlabel("UTM E (km)"); ax0.set_ylabel("UTM N (km)")

    im1 = ax1.imshow(argM, extent=ext, origin=org, cmap="RdBu_r",
                     vmin=-90, vmax=90)
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="arg(M) phase bias (deg)")
    ax1.set_title(f"arg(M)   (median {np.nanmedian(argM):+.1f} deg)")
    ax1.set_xlabel("UTM E (km)")

    fig.suptitle(f"{wy}: ASO sub-pixel M at {WINDOW_M} m, {band}-band "
                 f"({AVG_RES_M} m cells, kappa = {kappa:.1f} rad/m)", fontsize=14)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"aso_M_81m_{wy}_{band}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    kstr = ", ".join(f"{b}={kappa_of(lam):.1f}" for b, lam in BANDS.items())
    print(f"window = {WINDOW_M} m ({BLOCK} x {BLOCK} cells of {AVG_RES_M} m, "
          f"pre-averaged {PREAVG}x{PREAVG} from {BASE_RES_M} m)")
    print(f"kappa (rad/m): {kstr}\n")
    print(f"{'pair':8s} {'band':>5s} {'n_windows':>10s} {'median |M|':>11s} "
          f"{'median arg(M)':>14s}")
    for wy, date_a, date_b in PAIRS:
        # pre-average once per pair (band-independent), then M for each band
        dswe = preaverage(load_dswe(date_a, date_b))   # 3 m -> 9 m (noise)
        for band, lam in BANDS.items():
            kappa = kappa_of(lam)
            M = window_M(dswe, kappa)
            absM = np.abs(M.values)
            argM = np.degrees(np.angle(M.values))
            argM[~np.isfinite(absM)] = np.nan
            n = int(np.isfinite(absM).sum())

            save_M(M, pair_dir(date_a, date_b), band)
            plot_M(M, wy, band, kappa)
            print(f"{wy:8s} {band:>5s} {n:10,d} {np.nanmedian(absM):11.3f} "
                  f"{np.nanmedian(argM):+13.1f}")


if __name__ == "__main__":
    main()
