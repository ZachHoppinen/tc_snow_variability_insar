#!/usr/bin/env bash
# generate_nisar_products.sh -- build the NISAR products for the multi-resolution
# coherence check (the only NISAR analysis in the paper), from the staged raw
# inputs. Operates on the two delivered 12-day GUNWs (P12, P23); no SAS run.
#
#   1. calibrate_enl       single oversampling factor f + ENL_20/ENL_80 over water
#   2. debias_coherences   bias-corrected 20 & 80 m coherence (raw+corrected
#                          GeoTIFFs) for both snow-relevant pairs P12 and P23
#
# Step 1 only PRINTS the calibrated f/ENL for the record; step 2 reads its own
# F_OVERSAMPLE constant (keep debias_coherences.py F_OVERSAMPLE in sync with the
# value step 1 reports). Both run in nisar_pytools.
#
# The long-baseline P13 GUNW generation (prep_nisar.py + p13_template_runconfig.yaml)
# and the triplet closure phase (phase_closure.py) were for the closure analysis,
# which is not in the paper; they now live in nisar/exploratory/.
#
# Usage:
#   bash generate_nisar_products.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/miniforge3/envs/nisar_pytools/bin/python"
DATA="$HOME/Documents/nisar_swe/data/sub_pixel_variability"

step() { printf '\n========== %s ==========\n' "$1"; }

# 1. effective number of looks over open water (single oversampling factor)
step "1/2 calibrate_enl (ENL over water)"
"$PY" "$HERE/calibrate_enl.py"

# 2. bias-corrected 20-80 m coherence for both pairs
step "2/2 debias_coherences (P12, P23)"
for pair in P12 P23; do
    "$PY" "$HERE/debias_coherences.py" --pair "$pair"
done

step "done"
echo "products under: $DATA/processed/nisar/coherence"
