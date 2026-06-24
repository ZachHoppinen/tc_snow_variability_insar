"""constants_check.py -- confirm the Methods-section CONSTANTS in v4.tex against
the actual code constants, by importing them from their source modules.

The other stats scripts cover the computed statistics; this one covers the
fixed parameters quoted in the Methods (kappa values, window/block geometry,
look counts, oversampling factor, density-calibration thresholds). It imports
each constant from the single module that defines it -- so if a constant is
edited in the code, this check flags the mismatch with the manuscript.

Run:  conda activate nisar_pytools && python stats/constants_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ASO"))
sys.path.insert(0, str(HERE.parent / "nisar"))

# --- the single source of truth for each constant -------------------------
from aso_M_calculation import (  # noqa: E402  (ASO/)
    BANDS, BASE_RES_M, PREAVG, AVG_RES_M, WINDOW_M, BLOCK, MIN_FRAC,
    INC_RAD, kappa_of,
)
from aso_pairwise_dswe import MIN_DDEPTH_M, RHO_MIN, RHO_MAX, PAIRS  # noqa: E402
from calibrate_enl import N_LOOKS_20, N_LOOKS_80  # noqa: E402  (nisar/)
from debias_coherences import F_OVERSAMPLE, L_20, L_80  # noqa: E402


def check(label: str, text_value, code_value, tol=None) -> bool:
    """Print one row: manuscript value vs code value, with a match flag."""
    if tol is None:
        ok = text_value == code_value
    else:
        ok = abs(float(text_value) - float(code_value)) <= tol
    flag = "ok " if ok else "MISMATCH"
    print(f"  [{flag}] {label:42s} text={text_value!s:>10}  code={code_value!s:>10}")
    return ok


def main():
    print("Methods-section constants: manuscript value vs imported code value\n")
    allok = True

    print("M3.3  geometry / wavelength (ASO/aso_M_calculation.py)")
    allok &= check("incidence angle (deg)", 40.0, round(np.rad2deg(INC_RAD), 3))
    allok &= check("ASO base posting (m)", 3, BASE_RES_M)
    allok &= check("pre-average block -> working posting (m)", 9, AVG_RES_M)
    allok &= check("multilook window (m)", 81, WINDOW_M)
    allok &= check("cells per window side (BLOCK)", 9, BLOCK)
    nmin = int(np.ceil(MIN_FRAC * BLOCK * BLOCK))
    allok &= check("min valid cells N>=  (MIN_FRAC*BLOCK^2)", 41, nmin)
    for band, ktext in (("C", 224.1), ("L", 52.7), ("P", 15.5)):
        allok &= check(f"kappa {band}-band (rad/m, lambda={BANDS[band]} m)",
                       ktext, round(kappa_of(BANDS[band]), 1), tol=0.1)

    print("\nM3.2  density calibration thresholds (ASO/aso_pairwise_dswe.py)")
    allok &= check("min |delta-depth| (cm)", 10, int(MIN_DDEPTH_M * 100))
    allok &= check("density bounds (kg/m^3)", "[50, 600]",
                   f"[{int(RHO_MIN)}, {int(RHO_MAX)}]")
    print(f"        (per-pair densities 304/312/373/330 are COMPUTED at runtime "
          f"from {len(PAIRS)} pairs -> run stats/../ASO/aso_pairwise_dswe.py)")

    print("\nM3.4  NISAR multilook bias (nisar/calibrate_enl.py, debias_coherences.py)")
    allok &= check("nominal looks 20 m (5x6)", 30, N_LOOKS_20)
    allok &= check("nominal looks 80 m (13x16)", 208, N_LOOKS_80)
    allok &= check("oversampling factor f", 1.29, round(F_OVERSAMPLE, 2), tol=0.005)
    allok &= check("effective looks L_20 = N/f", 23, round(L_20))
    allok &= check("effective looks L_80 = N/f", 162, round(L_80))

    print("\n" + ("ALL CONSTANTS MATCH THE MANUSCRIPT" if allok
                  else "*** SOME CONSTANTS DIFFER -- reconcile code and text ***"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
