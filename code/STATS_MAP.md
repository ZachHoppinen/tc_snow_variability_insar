# v4 stats provenance map

Every quantitative claim in `v4.tex` and the script that generates it. Run any
script from the repo with the `nisar_pytools` env. Section numbers refer to the
manuscript headings. Confirmed = numbers reproduced and matched to the text.

Core dependency: all ASO `|M|` stats import the M pipeline + constants
(`kappa_of`, `PREAVG`, `BLOCK`, `MIN_FRAC`, band wavelengths, grid origin) from
`ASO/aso_M_calculation.py`. Per-pair densities come from `ASO/aso_pairwise_dswe.py`.

| § | Stat in text | Script | Status |
|---|---|---|---|
| **M3.1** | Dec 7-19 mean -1 cm, Dec 19-31 mean +18 cm SWE (CDEC pillows) | `stats/interval_accumulation.py` | maps (CDEC fetch) |
| **M3.2** | per-pair densities 304/312/373/330 kg m^-3; 10 cm + [50,600] filter | `ASO/aso_pairwise_dswe.py` (thresholds in `stats/constants_check.py`) | **confirmed** |
| **M3.3** | 3x3->9 m; 9x9=81 m; N>=41; kappa 224.1/52.7/15.5; 40 deg | `ASO/aso_M_calculation.py` -> `stats/constants_check.py` | **confirmed** |
| **M3.4** | looks 30 (20 m) / 208 (80 m); f~1.29; L20~23, L80~162; 13,189 water samples | `nisar/calibrate_enl.py`,`debias_coherences.py` -> `stats/constants_check.py` | **confirmed** (calibrate_enl rerun: 13,189, f=1.287) |
| **R4.1** | 1.34 M windows; dSWE med 12.6 (IQR 4.7-20.4); within-win sigma 3.4 (2.1-6.3); skew med +0.07 mean +0.14 | `stats/aso_M_stats.py` | confirmed (skew/N) |
| **R4.1** | median \|M\| 0.25-0.47 (IQR 0.15-0.58); bias <=2 deg, IQR +/-16, -17/+30 WY2024; storms 0.25-0.31 -> 0.12-0.16 steep; WY2023 sigma 2.8, \|M\| 0.47 | `stats/aso_M_stats.py` | maps |
| **R4.1** | per-pair vs four-pair-mean r^2: \|M\| 0.75-0.82, \|arg(M)\| 0.59-0.63; signed arg(M) pairwise r^2 < 0.01 | `stats/consistency_stats.py` | **confirmed** |
| **subsec:psd / R(PSD)** | per-window dSWE PSD break: median 27 m (IQR 22-31, 570 windows), small-scale beta 2.6 / large-scale beta 1.5, 88% in Deems 15-40 m, per-pair 22-27 m, 0% at scan floor | `stats/psd_break_stats.py` (mode-weighted WLS, amplitude-normalized, per-pair) | **confirmed**; in v4.tex |
| **R(window sizes)** | pooled median \|M\| 0.54 at 9 m / 0.33 at ~20 m / 0.18 at 81 m; IQR 0.07-0.38 at 81 m; bias med <=0.3 mm, IQR ~+/-2 mm (\|M\|>0.3) | `stats/window_size_stats.py` (uses `exploratory/window_size_sweep.py` helpers) | **confirmed** (0.542 / 0.348@18m-0.319@21m / 0.183; IQR 0.070-0.381; bias med max +0.26 mm, IQR -1.2 to +2.3 mm) |
| **R4.2** | C \|M\| 0.10 (0.06-0.14) 53% <0.1, arg -86/+88; L 0.31 (0.15-0.55) 14%, arg -15/+19; P 0.87 (0.65-0.94) <1%, arg -1.6/+0.3; pooled 0.10/0.33/0.88 | `stats/stats_frequencies.py` | **confirmed** |
| **R4.3** | corrected dgamma over the common 4-pair snow footprint: P12 med -0.004 (45% +), P23 +0.031 (71% +); raw gap P23 +0.072; p90 ~0.13, g80 0.17 (high) vs 0.27 (low) | `stats/coherence_resolution_stats.py` (4-pair intersection mask) | **confirmed** |
| **D4.1** | melt \|M\| 0.55 vs accum 0.32; within-win sigma 2.4 vs 3.7; matched-magnitude | `stats/melt_zone_stats.py` | confirmed (session) |
| **D4.5** | constant vs variable density: \|M\| shift <=0.03, Pearson r 0.91-0.98, arg <2 deg | `stats/stats_constant_density.py` | confirmed (session) |
| **D4.5** | noise: PSD 1.7 cm/3 m -> 0.5 cm/9 m; bare ground 1.6 cm/3 m; noise-only \|M\| 0.45/0.96/1.00 (C/L/P) | `stats/noise_methods_compare.py` | confirmed (session) |
| **App. B** | VIIRS VNP10A1F snow cover vs corrected dgamma (WY2023 footprint): P12 r +0.10 / rho +0.12 (snow 11%); P23 r +0.35 / rho +0.40 (snow 23%); footprint fully snow-covered | `stats/viirs_snowcover_stats.py` -> fig `visualizations/plot_viirs_snowcover.py` (B1) | **confirmed** (session) |

## Figure scripts (visualizations/)

| Figure in text | Script |
|---|---|
| `teach_phasor_contrast.png` (theory) | `visualizations/teach_phasor_contrast.py` |
| `fig_phase_wraps_bands.png` (M3.3) | `visualizations/phase_wraps_bands.py` |
| `all_pairs_dswe_M_L.png` (R4.1) | `visualizations/plot_all_pairs.py` |
| `bands_M_WY2025.png` (R4.2) | `visualizations/plot_bands_one_pair.py` |
| `coherence_rows_P12_P23.png` (R4.3) | `visualizations/plot_coherence_rows.py` (reads `nisar/debias_coherences.py` output) |
| `dswe_psd_break.png` (fig:psd_break, R PSD section) | `visualizations/plot_dswe_psd_break.py` (imports `compute_window_breaks()` from `stats/psd_break_stats.py`, only plots) |
| `window_M_arg_vs_size_L.png` (fig:window_size, R window-size section) | `visualizations/plot_window_M_arg.py` (stats source: `stats/window_size_stats.py`) |

## stats/ layout

`stats/` holds only the manuscript-load-bearing scripts (the table above) plus:
- `stats/constants_check.py` -- imports + verifies the Methods constants.
- `stats/aso_noise.py` -- NOT cited as a stat itself (the calm-region noise
  estimate was superseded), but it is a **dependency** of the cited
  `noise_methods_compare.py`, which imports `calm_mask`, `coarse_window_fields`,
  and `detrended_noise` from it. Kept in `stats/` as a helper.

## stats/exploratory/ -- not cited in the text

Moved out of `stats/` so the top level is only manuscript stats. Import paths
fixed for the extra directory level.

| Script | Why it exists | Disposition |
|---|---|---|
| `exploratory/aso_noise_preavg_test.py` | pre-average sensitivity check for the noise method | exploratory |
| `exploratory/plot_calm_region.py` | figure of the rejected calm-region selection | superseded |
| `exploratory/plot_bareground_map.py` | bare-ground diagnostic maps supporting the D4.5 noise bound (no text number) | supporting figure |
| `exploratory/window_size_sweep.py` | helper module (M_for_side, common mask, loaders) imported by the cited `stats/window_size_stats.py` | dependency, keep |
| `exploratory/closure_stats.py` | triplet closure (discussed hypothetically in D4.4/future; not a cited stat) | future |
| `exploratory/coherence_vs_M.py` | NISAR-vs-\|M\| spatial agreement (decided not to use) | dropped |

## nisar/ layout

The paper's only NISAR analysis is the multi-resolution coherence check (P12/P23),
so `nisar/` holds just that pipeline:
- `calibrate_enl.py` (M3.4 f/ENL), `debias_coherences.py` (R4.3 corrected coherence),
  `build_calibration_water_mask.py` (water mask), `generate_nisar_products.sh` (driver).

`nisar/exploratory/` -- closure-only, not in the paper:
- `prep_nisar.py` + `p13_template_runconfig.yaml` -- long-baseline P13/P24 GUNW
  generation (the triplet third leg).
- `phase_closure.py` -- triplet closure phase.

Related orphan still in `visualizations/`: `plot_closure_vs_argM.py` (closure figure;
reads `phase_closure.py` output). Not a cited figure.
