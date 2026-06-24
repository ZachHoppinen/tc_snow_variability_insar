"""aso_M_stats.py -- per-pair / pooled statistics of the 81 m M analysis, for
writing up the four ASO pairs.

For each pair we reproduce the M pipeline exactly (3 m dSWE -> 3x3 block-average
to 9 m -> non-overlapping 9x9 = 81 m windows, N >= 41 valid cells) and, in one
pass per window, record four things:

  - window-mean dSWE     (cm)      the SIGNAL MAGNITUDE in that window
  - within-window sigma  (cm)      the sub-pixel VARIABILITY (std of the 9 m
                                   cells inside the window) -- this is what
                                   drives |M|
  - |M|                            coherence factor  (from the same window)
  - arg(M)               (deg)     phase bias

We then report, per water year and pooled over all four pairs, the mean and the
IQR (p25, p75) of each. Comparing the within-window sigma across years vs the
window-mean dSWE across years answers: does the sub-pixel VARIABILITY change
year to year, or only the MAGNITUDE of the signal (with similar variability)?
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-slice windows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ASO"))
sys.path.insert(0, str(HERE.parent))  # repo code/ dir
from aso_M_calculation import kappa_of, PREAVG, BLOCK, MIN_FRAC, BANDS  # noqa: E402
from tc_paths import DATA_ROOT  # noqa: E402

PROCESSED_DIR = DATA_ROOT / "sub_pixel_variability/processed/aso"
DEM_TIF = DATA_ROOT / "sub_pixel_variability/raw/nisar/dem.tif"
PAIRS = [
    ("WY2023", "2023Mar02-03",    "2023Mar16-17"),
    ("WY2024", "2024Jan29",       "2024Feb27-28"),
    ("WY2025", "2025Feb08-09",    "2025Feb25"),
    ("WY2026", "2026Jan31-Feb01", "2026Feb27-28"),
]
KAPPA = kappa_of(BANDS["L"])          # L-band, the analysis band
MIN_CELLS = int(np.ceil(MIN_FRAC * BLOCK * BLOCK))   # >= 41 of 81


def elev_on(date_a: str, date_b: str) -> np.ndarray:
    """Elevation (m) on the pair's 81 m window grid. The DEM (EPSG:4326) is
    reprojected bilinearly onto the M netCDF grid (which is the window grid);
    only the DEM is resampled."""
    pdir = PROCESSED_DIR / f"{date_a}_{date_b}"
    with xr.open_dataset(pdir / "aso_M_81m_L.nc") as ds:
        arr = ds["M_real"]
        tmpl = xr.DataArray(np.zeros(arr.shape, "float32"),
                            dims=arr.dims, coords=arr.coords)
    tmpl.rio.write_crs("EPSG:32611", inplace=True)
    with rioxarray.open_rasterio(DEM_TIF, masked=True) as d0:
        dem = d0.squeeze(drop=True).load()
    return dem.rio.reproject_match(tmpl, resampling=Resampling.bilinear).values


def window_stats(date_a: str, date_b: str):
    """Return per-window arrays (mean_dswe_cm, within_sigma_cm, within_skew,
    absM, argM_deg, elev_m) over valid 81 m windows for one pair. The skew is
    the standardized 3rd moment of the 9 m dSWE within each window
    (dimensionless; same value for unwrapped phase since dphi = kappa*dSWE)."""
    pdir = PROCESSED_DIR / f"{date_a}_{date_b}"
    with xr.open_dataset(pdir / "aso_dswe_3m.nc") as ds:
        d3 = ds["dswe"].load()
    # 3 m -> 9 m block-average (matches the M pipeline pre-average)
    d9 = d3.coarsen(x=PREAVG, y=PREAVG, boundary="trim").mean(skipna=True).values

    ny, nx = d9.shape
    nby, nbx = ny // BLOCK, nx // BLOCK
    d = d9[:nby * BLOCK, :nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)  # m
    valid = np.isfinite(d)
    cnt = valid.sum(axis=(1, 3))

    with np.errstate(invalid="ignore"):
        mean_d = np.nanmean(d, axis=(1, 3))             # m
        dev = d - mean_d[:, None, :, None]
        std_d = np.sqrt(np.nanmean(dev ** 2, axis=(1, 3)))   # m (within-window spread)
        skew = np.nanmean(dev ** 3, axis=(1, 3)) / std_d ** 3  # standardized 3rd moment
    z = np.where(valid, np.exp(1j * KAPPA * np.nan_to_num(d)), 0.0)
    zbar = z.sum(axis=(1, 3)) / np.maximum(cnt, 1)
    M = zbar * np.exp(-1j * KAPPA * np.nan_to_num(mean_d))

    elev = elev_on(date_a, date_b)          # (nby, nbx), same grid as the M output
    assert elev.shape == cnt.shape, (elev.shape, cnt.shape)

    ok = (cnt >= MIN_CELLS) & np.isfinite(skew)
    return (mean_d[ok] * 100.0, std_d[ok] * 100.0, skew[ok],
            np.abs(M)[ok], np.degrees(np.angle(M))[ok], elev[ok])


def summarize(name: str, arr: np.ndarray) -> str:
    p25, p75 = np.percentile(arr, [25, 75])
    return (f"{name:9s} mean={arr.mean():8.3f}  median={np.median(arr):8.3f}  "
            f"IQR=[{p25:7.3f}, {p75:7.3f}]  (n={arr.size:,})")


def main() -> None:
    cols = {"mean_dswe": "window-mean dSWE (cm)  [signal magnitude]",
            "sigma": "within-window sigma dSWE (cm)  [sub-pixel variability]",
            "skew": "within-window dSWE skewness  [standardized 3rd moment]",
            "absM": "|M|  (coherence factor)",
            "argM": "arg(M) (deg)  [phase bias]",
            "bias_mm": "SWE-equivalent bias (mm)  [arg(M)/kappa]"}
    data = {k: {} for k in cols}            # data[col][wy] = array
    data["elev"] = {}

    for wy, a, b in PAIRS:
        md, sd, sk, am, ag, ev = window_stats(a, b)
        data["mean_dswe"][wy] = md
        data["sigma"][wy] = sd
        data["skew"][wy] = sk
        data["absM"][wy] = am
        data["argM"][wy] = ag
        # arg(M) (deg) -> SWE bias: dphi = kappa*dSWE, so dSWE = arg(M)/kappa
        data["bias_mm"][wy] = np.deg2rad(ag) / KAPPA * 1000.0
        data["elev"][wy] = ev

    for col, title in cols.items():
        print(f"\n=== {title} ===")
        for wy, *_ in PAIRS:
            print("  " + summarize(wy, data[col][wy]))
        pooled = np.concatenate([data[col][wy] for wy, *_ in PAIRS])
        print("  " + summarize("POOLED", pooled))

    # the variability-vs-magnitude question, side by side
    print("\n=== variability vs magnitude, year to year ===")
    print(f"  {'WY':7s} {'mean dSWE':>10s} {'within-win sigma':>17s} "
          f"{'median |M|':>11s}")
    for wy, *_ in PAIRS:
        print(f"  {wy:7s} {data['mean_dswe'][wy].mean():10.2f} "
              f"{data['sigma'][wy].mean():17.2f} "
              f"{np.median(data['absM'][wy]):11.3f}")

    # where the bias lives: arg(M) / SWE-bias by elevation quintile (pooled).
    # Reported as MAGNITUDE (|arg(M)|, |bias|) since the per-window bias is
    # near-symmetric in sign; the claim is that its size grows in the high,
    # wind-exposed alpine windows where |M| (coherence) is lowest.
    elev = np.concatenate([data["elev"][wy] for wy, *_ in PAIRS])
    absarg = np.abs(np.concatenate([data["argM"][wy] for wy, *_ in PAIRS]))
    absbias = np.abs(np.concatenate([data["bias_mm"][wy] for wy, *_ in PAIRS]))
    aM = np.concatenate([data["absM"][wy] for wy, *_ in PAIRS])
    sg = np.concatenate([data["sigma"][wy] for wy, *_ in PAIRS])
    print("\n=== pooled |arg(M)| / |bias| by elevation quintile ===")
    print(f"  {'elev band (m)':>16s} {'med|argM|':>10s} {'med|bias|mm':>12s} "
          f"{'p90|bias|mm':>12s} {'med|M|':>8s} {'med sigma':>10s}")
    edges = np.percentile(elev, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        sel = (elev >= lo) & (elev <= hi if i == 4 else elev < hi)
        print(f"  {lo:6.0f}-{hi:6.0f}    {np.median(absarg[sel]):10.2f} "
              f"{np.median(absbias[sel]):12.2f} {np.percentile(absbias[sel],90):12.2f} "
              f"{np.median(aM[sel]):8.3f} {np.median(sg[sel]):10.2f}")

    # lowest-|M| (worst-coherence) decile: largest within-window variability,
    # hence the largest-magnitude bias -- and the least reliable phase.
    print("\n=== lowest-|M| decile (worst coherence), pooled ===")
    thr = np.percentile(aM, 10)
    lo = aM <= thr
    print(f"  |M| <= {thr:.3f} (n={lo.sum():,}): "
          f"median |arg(M)|={np.median(absarg[lo]):.2f} deg, "
          f"median |bias|={np.median(absbias[lo]):.2f} mm, "
          f"p90 |bias|={np.percentile(absbias[lo],90):.2f} mm, "
          f"median sigma={np.median(sg[lo]):.2f} cm")

    # per-pair high-alpine (top elevation quintile) SWE bias -- sources the
    # "~1 cm in the most wind-affected alpine windows of the largest event".
    print("\n=== high-alpine windows (top elevation quintile), per pair ===")
    for wy, *_ in PAIRS:
        ev, bm = data["elev"][wy], np.abs(data["bias_mm"][wy])
        hi = ev >= np.percentile(ev, 80)
        print(f"  {wy}: top-quintile (>={np.percentile(ev,80):.0f} m, n={hi.sum():,})  "
              f"median |bias|={np.median(bm[hi]):.2f} mm  "
              f"p90={np.percentile(bm[hi],90):.2f} mm  "
              f"max={bm[hi].max():.2f} mm")


if __name__ == "__main__":
    main()
