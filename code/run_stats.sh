#!/usr/bin/env bash
# Regenerate / confirm every statistic cited in v4.tex, grouped by manuscript
# section. See STATS_MAP.md for the section -> script -> stat mapping.
# Usage:  conda activate nisar_pytools && bash run_stats.sh   [section]
#   no arg     run all
#   3.1 / 4.2  run only that section's script(s)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
SEC="${1:-all}"

banner() { printf '\n========================================================\n%s\n========================================================\n' "$1"; }
run() { banner "$1"; echo "(script: $2)"; python "$2"; }

want() { [ "$SEC" = "all" ] || [ "$SEC" = "$1" ]; }

want const && run "M3.2-3.4 constants check (kappa, looks, f, geometry)"        stats/constants_check.py
want 3.1 && run "M3.1  NISAR interval accumulation (CDEC pillows: -1 / +18 cm)" stats/interval_accumulation.py
want 3.1 && run "M3.1  NISAR baselines, height of ambiguity, geometric coherence" stats/nisar_baselines.py
want 3.4 && run "M3.4  NISAR Bamler oversample factor f, L20, L80"              nisar/calibrate_enl.py
want 4.1 && run "R4.1  ASO |M|/arg(M) per-pair + pooled stats"                  stats/aso_M_stats.py
want 4.1 && run "R4.1  spatial consistency r^2 across pairs"                    stats/consistency_stats.py
want 4.1 && run "subsec:psd  per-window dSWE PSD two-regime break (~27 m)"      stats/psd_break_stats.py
want 4.2 && run "R4.2  per-band (C/L/P) |M| and arg(M)"                         stats/stats_frequencies.py
want 4.4 && run "R(window sizes)  |M| and bias vs multilook footprint (9-81 m)" stats/window_size_stats.py
want 4.3 && run "R4.3  NISAR multi-resolution coherence (dgamma)"              stats/coherence_resolution_stats.py
want B && run "Appendix B  VIIRS snow cover vs dgamma correlation"             stats/viirs_snowcover_stats.py
want 4.1d && run "D4.1 melt vs accumulation |M| (WY2023)"                       stats/melt_zone_stats.py
want 4.5 && run "D4.5  constant vs variable density"                            stats/stats_constant_density.py
want 4.5 && run "D4.5  ASO depth-noise floor (PSD + bare ground)"              stats/noise_methods_compare.py

banner "done"
