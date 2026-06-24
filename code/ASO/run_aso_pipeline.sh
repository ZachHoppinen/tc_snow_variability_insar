#!/usr/bin/env bash
# Full ASO sub-pixel-variability pipeline: raw flight zips -> dSWE -> M.
#
# Runs the three stages in order, stopping at the first failure:
#   1. prep_aso.py          unzip raw flight archives into per-pair folders
#   2. aso_pairwise_dswe.py difference depths, calibrate density, 3 m dSWE map
#   3. aso_M_calculation.py complex sub-pixel M factor per 81 m window
#
# All I/O paths are absolute (set inside each script), so this just sequences
# them. Activate the environment first, e.g.:  conda activate nisar_pytools
#
# Override the interpreter if needed:  PYTHON=python3.11 ./run_aso_pipeline.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_stage () {
    local label="$1" script="$2"
    echo
    echo "=================================================================="
    echo ">> ${label}  (${script})"
    echo "=================================================================="
    "${PYTHON}" "${HERE}/${script}"
}

run_stage "1/3  prep (unzip raw ASO)"        prep_aso.py
run_stage "2/3  pairwise differencing -> dSWE" aso_pairwise_dswe.py
run_stage "3/3  per-window complex M"        aso_M_calculation.py

echo
echo "ASO pipeline complete."
