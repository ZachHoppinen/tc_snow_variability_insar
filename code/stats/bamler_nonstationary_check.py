"""bamler_nonstationary_check.py -- does the Bamler finite-look bias correction
(Eq. bamler, Section 3.6) still hold when the L looks do not share a phase?
(referee 2: validity of the stationary-sample expectation under non-stationarity)
 Simulate L correlated circular-Gaussian look pairs with
speckle coherence gamma_other and a deterministic per-look phase offset whose
mean phasor has magnitude |M|. Compare the Monte Carlo mean sample coherence
with Bamler's E{ghat | gamma_other*|M|, L}, and see what the paper's inverse
(de-bias) recovers relative to the effective coherence gamma_other*|M|."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nisar"))
from calibrate_enl import bamler_mean, forward_curve
from scipy.interpolate import interp1d
rng = np.random.default_rng(0)
def sample_coh(L, g, phases, ntrial=4000):
    a = (rng.standard_normal((ntrial, L)) + 1j * rng.standard_normal((ntrial, L))) / np.sqrt(2)
    b = (rng.standard_normal((ntrial, L)) + 1j * rng.standard_normal((ntrial, L))) / np.sqrt(2)
    z1 = a; z2 = g * a * np.exp(1j * phases) + np.sqrt(1 - g**2) * b
    num = np.abs((z1 * np.conj(z2)).sum(1)); den = np.sqrt((np.abs(z1)**2).sum(1) * (np.abs(z2)**2).sum(1))
    return (num / den).mean()
def phases_for_M(L, M):
    # two-level pattern: half the looks at +a, half at -a -> |M| = cos(a)
    a = np.arccos(M); ph = np.where(np.arange(L) % 2 == 0, a, -a); return ph
print(f"{'L':>4s} {'g_other':>7s} {'|M|':>5s} {'g_eff':>6s} | {'MC E[ghat]':>10s} {'Bamler(g_eff)':>13s} {'diff':>6s} | {'debiased MC':>11s} {'err vs g_eff':>12s}")
for L in (23, 158):
    g, gh = forward_curve(L, n=40, gmax=0.9); deb = interp1d(gh, g, bounds_error=False, fill_value=(0, 0.9))
    for g in (0.3, 0.5, 0.8):
        for M in (1.0, 0.7, 0.4):
            geff = g * M
            mc = sample_coh(L, g, phases_for_M(L, M))
            bam = bamler_mean(geff, L)
            print(f"{L:4d} {g:7.2f} {M:5.2f} {geff:6.3f} | {mc:10.3f} {bam:13.3f} {mc-bam:+6.3f} | {float(deb(mc)):11.3f} {float(deb(mc))-geff:+12.3f}")
