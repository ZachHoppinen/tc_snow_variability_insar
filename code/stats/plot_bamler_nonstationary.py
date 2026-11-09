"""plot_bamler_nonstationary.py -- recovered vs true effective coherence as the
within-window phase spread grows (response to referee 2 on the validity of the
Bamler de-bias, Eq. bamler, under non-stationary looks).

For L correlated looks with speckle coherence gamma_other and a Gaussian per-look
phase pattern of std sigma: the true effective coherence of each window,
|M_window| * gamma_other with M_window the mean phasor of that window's own L
phases (averaged over windows), against the Bamler-de-biased sample coherence
the pipeline would report, with the raw sample coherence for reference. Companion to bamler_nonstationary_check.py,
which uses a deterministic two-level phase pattern instead of a random Gaussian one.

Run from the repo root in the nisar_pytools env:
    python code/stats/plot_bamler_nonstationary.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nisar")); sys.path.insert(0, str(HERE.parent))
from calibrate_enl import forward_curve
from tc_paths import FIG_ROOT
rng = np.random.default_rng(0)
def sample_coh(L, g, sigma, ntrial=3000):
    ph = rng.normal(0, sigma, (ntrial, L))                       # a new phase pattern per window
    a = (rng.standard_normal((ntrial, L)) + 1j * rng.standard_normal((ntrial, L))) / np.sqrt(2)
    b = (rng.standard_normal((ntrial, L)) + 1j * rng.standard_normal((ntrial, L))) / np.sqrt(2)
    z1 = a; z2 = g * a * np.exp(1j * ph) + np.sqrt(1 - g**2) * b
    num = np.abs((z1 * np.conj(z2)).sum(1)); den = np.sqrt((np.abs(z1)**2).sum(1) * (np.abs(z2)**2).sum(1))
    m_win = np.abs(np.exp(1j * ph).mean(1))                       # each window's own |M|
    return (num / den).mean(), (g * m_win).mean()
sig = np.linspace(0, 2.5, 14)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
for ax, L in zip(axes, (23, 158)):
    g_, gh_ = forward_curve(L, n=40, gmax=0.95); deb = interp1d(gh_, g_, bounds_error=False, fill_value=(0, 0.95))
    for g, col in ((0.8, "tab:blue"), (0.5, "tab:orange"), (0.2, "tab:green")):
        res = np.array([sample_coh(L, g, s) for s in sig]); raw, truth = res[:, 0], res[:, 1]; rec = deb(raw)
        ax.plot(sig, truth, color=col, lw=2, label=fr"true $|M|\gamma_\mathrm{{other}}$, $\gamma_\mathrm{{other}}$ = {g}")
        ax.plot(sig, raw, color=col, ls=":", lw=1.5, label="raw sample coherence")
        ax.plot(sig, rec, color=col, ls="none", marker="o", ms=5, label="de-biased sample coherence")
        print(f"L={L} g={g}: max |de-biased - true| = {np.max(np.abs(rec - truth)):.3f}")
    ax.set_xlabel("within-window phase spread (rad)"); ax.set_title(f"{L} looks")
    ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="upper right")
axes[0].set_ylabel("coherence")
out = FIG_ROOT / "stats" / "bamler_nonstationary.png"; out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150); print("wrote", out)
