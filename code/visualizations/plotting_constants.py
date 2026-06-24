"""plotting_constants.py -- shared styling, colors, and paths for the v4
visualization figures.

Every plot_*.py imports what it needs from here so that fonts, colormaps,
color limits, per-pair and per-band colors, and output paths stay consistent
across panels. Call apply_style() once at the top of main() before plotting.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# pull the repo-relative roots (code/ is one level up from visualizations/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tc_paths import DATA_ROOT, FIG_ROOT  # noqa: E402

# --- shared paths -----------------------------------------------------------
PROCESSED_DIR = DATA_ROOT / "sub_pixel_variability/processed/aso"
FIG_DIR = FIG_ROOT / "visualizations"

# --- the four ASO pairs (water-year label, early flight, late flight) -------
PAIRS = [
    ("WY2023", "2023Mar02-03",    "2023Mar16-17"),
    ("WY2024", "2024Jan29",       "2024Feb27-28"),
    ("WY2025", "2025Feb08-09",    "2025Feb25"),
    ("WY2026", "2026Jan31-Feb01", "2026Feb27-28"),
]
PAIR_COLORS = {"WY2023": "#8c564b", "WY2024": "#9467bd",
               "WY2025": "#7f7f7f", "WY2026": "#e377c2"}

# --- per-band colors and labels (C / L / P) ---------------------------------
BAND_COLORS = {"C": "#80b4d6", "L": "#2ca02c", "P": "#d62728"}
BAND_LABELS = {"C": r"C-band ($\lambda$ = 5.6 cm)",
               "L": r"L-band ($\lambda$ = 23.8 cm)",
               "P": r"P-band ($\lambda$ = 81 cm)"}

# --- typography (one source for all figures) --------------------------------
RC = {"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
      "xtick.labelsize": 12, "ytick.labelsize": 12,
      "legend.fontsize": 12, "figure.titlesize": 18}
DPI = 150


def apply_style() -> None:
    """Apply the shared rcParams. Call once before plotting."""
    plt.rcParams.update(RC)


# --- colormaps (one meaning each, used across figures) ----------------------
CMAP_COH = "viridis"        # |M| and observed coherence, 0..1
CMAP_ARG = "RdBu_r"         # arg(M) / phase bias, symmetric about 0
CMAP_DSWE = "RdBu"          # dSWE, symmetric about 0
CMAP_COHDIFF = "PuOr_r"     # bias-corrected gamma_20 - gamma_80, symmetric
CMAP_NISAR_COH = "Greys_r"  # observed NISAR coherence, 0..1
CMAP_TEACH = "Blues"        # schematic dSWE field / phasor colours

# --- shared color and value limits ------------------------------------------
COH_VMIN, COH_VMAX = 0.0, 1.0    # |M| / coherence
ARG_VLIM_DEG = 90.0              # symmetric color limit for arg(M) maps (deg)

# --- line colors for the |M| / arg(M) curves --------------------------------
COLOR_M = "tab:blue"             # coherence factor |M|
COLOR_ARG = "tab:red"            # phase bias arg(M)
