"""cumulant_check.py -- how well the cumulant expansion of M predicts |M| and
arg(M) window by window (Section 2, Eq. cumulant; referee 2).

  |M|    ~ exp(-s^2/2)                second order
  |M|    ~ exp(-s^2/2 + K s^4/24)     fourth order, K = excess kurtosis
  arg(M) ~ -S s^3/6                   third order, S = skew

with s the within-window phase spread. Reproduces the 81 m M pipeline (3 m
dSWE -> 9 m cells -> 9x9 windows) on every complete window of the four ASO
pairs and reports, per spread bin, the median absolute error of each
approximation against the true M, and for the usable windows (|M| > 0.3) the
arg error and the correlation of predicted and observed arg. Sources the
"within 0.05 and 3 deg up to 1.5 rad, fails beyond 1.75 rad" statement.

Run from the repo root in the nisar_pytools env with TC_DATA_ROOT set:
    python code/stats/cumulant_check.py
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ASO"))
from aso_M_stats import PAIRS, KAPPA, PROCESSED_DIR
from aso_M_calculation import PREAVG, BLOCK
S, SK, KU, AM, AG = [], [], [], [], []
for wy, a, b in PAIRS:
    with xr.open_dataset(PROCESSED_DIR / f"{a}_{b}" / "aso_dswe_3m.nc") as ds:
        d3 = ds["dswe"].load()
    d9 = d3.coarsen(x=PREAVG, y=PREAVG, boundary="trim").mean(skipna=True).values
    ny, nx = d9.shape; nby, nbx = ny // BLOCK, nx // BLOCK
    d = d9[:nby*BLOCK, :nbx*BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    valid = np.isfinite(d); cnt = valid.sum(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        m = np.nanmean(d, axis=(1, 3)); dev = d - m[:, None, :, None]
        sd = np.sqrt(np.nanmean(dev**2, axis=(1, 3)))
        sk = np.nanmean(dev**3, axis=(1, 3)) / sd**3
        ku = np.nanmean(dev**4, axis=(1, 3)) / sd**4 - 3
    z = np.where(valid, np.exp(1j*KAPPA*np.nan_to_num(d)), 0.0)
    M = z.sum(axis=(1, 3)) / np.maximum(cnt, 1) * np.exp(-1j*KAPPA*np.nan_to_num(m))
    ok = (cnt == 81) & np.isfinite(sk)
    S.append(KAPPA*sd[ok]); SK.append(sk[ok]); KU.append(ku[ok]); AM.append(np.abs(M)[ok]); AG.append(np.angle(M)[ok])
s, sk, ku, am, ag = map(np.concatenate, (S, SK, KU, AM, AG))
p2 = np.exp(-s**2/2); p4 = np.exp(-s**2/2 + ku*s**4/24); pa = -sk*s**3/6
print(f"n = {s.size:,} complete windows;  skew IQR {np.percentile(sk,[25,75]).round(2)}, excess kurtosis IQR {np.percentile(ku,[25,75]).round(2)}")
print(f"{'sigma':>9s} {'n':>8s} | {'|M| med':>7s} {'err2 med':>8s} {'err4 med':>8s} | {'|arg| med':>9s} {'arg err med':>11s} {'r(arg)':>6s}   [usable only for arg]")
for lo, hi in zip(np.arange(0, 3.0, 0.25), np.arange(0.25, 3.25, 0.25)):
    b = (s >= lo) & (s < hi)
    if b.sum() < 100: continue
    e2 = np.abs(p2[b] - am[b]); e4 = np.abs(p4[b] - am[b])
    u = b & (am > 0.3)
    ea = np.degrees(np.abs(pa[u] - ag[u])); r = np.corrcoef(pa[u], ag[u])[0, 1] if u.sum() > 2 else np.nan
    print(f"{lo:4.2f}-{hi:4.2f} {b.sum():8,} | {np.median(am[b]):7.2f} {np.median(e2):8.3f} {np.median(e4):8.3f} | "
          f"{np.degrees(np.median(np.abs(ag[u]))):9.1f} {np.median(ea):11.1f} {r:6.2f}")
