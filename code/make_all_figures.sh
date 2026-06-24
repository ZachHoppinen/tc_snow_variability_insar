#!/usr/bin/env bash
# Regenerate all v4 figures.
#
# Assumes the ASO pipeline has already produced the per-pair dSWE and M
# products (run ASO/run_aso_pipeline.sh first). This script only renders
# figures from those products; it stops at the first failure.
#
# Activate the environment first, e.g.:  conda activate nisar_pytools
# Override the interpreter if needed:     PYTHON=python3.11 ./make_all_figures.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run () {
    echo
    echo "=================================================================="
    echo ">> ${1}"
    echo "=================================================================="
    "${PYTHON}" "${HERE}/${1}"
}

# theory / schematic figures
run visualizations/teach_phasor_contrast.py
run visualizations/phase_wraps_bands.py

# data visualizations
run visualizations/plot_all_pairs.py
run visualizations/plot_bands_one_pair.py
run visualizations/plot_coherence_rows.py
run visualizations/plot_window_M_arg.py
run visualizations/plot_dswe_psd_break.py

echo
echo "all v4 figures complete."
