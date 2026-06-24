# Snow accumulation variability limits InSAR SWE retrieval

Companion code and manuscript for the study of how sub-resolution snow
accumulation (ΔSWE) variability biases and decorrelates L-band InSAR
snow-water-equivalent retrievals (NISAR-relevant), evaluated over the
Tuolumne basin using Airborne Snow Observatory (ASO) lidar.

## Repository layout

```
code/
  tc_paths.py       repo-relative data/figure roots (override via env vars)
  ASO/              ASO lidar -> per-pair ΔSWE and the per-window factor M
  nisar/            ENL calibration + coherence debiasing from delivered GUNWs
    download/       fetch the GUNWs (and RSLCs) from ASF
  stats/            statistics reported in the paper
  visualizations/   the figure scripts (plus shared plotting_constants.py)
  cache/            small WorldCover water mask (a required NISAR-tier input)
  make_all_figures.sh        regenerate the manuscript figures
  run_stats.sh               regenerate the reported statistics
  ASO/run_aso_pipeline.sh    raw ASO zips -> ΔSWE -> M
  nisar/generate_nisar_products.sh   ENL calibration + debiased coherence
  STATS_MAP.md               maps each reported number to the script that makes it
manuscript/   self-contained Copernicus LaTeX source (main.tex) + figures
figures/      generated output figures (so results are viewable without the data)
environment.yml   conda environment used for the analysis
```

## Paths: where data and figures live

All scripts resolve their inputs and outputs from two roots defined in
`code/tc_paths.py`, which default to `data/` and `figures/` inside the repo:

- `DATA_ROOT`  — input data       (default `./data`,    override `TC_DATA_ROOT`)
- `FIG_ROOT`   — figure output     (default `./figures`, override `TC_FIG_ROOT`)

Point them at a local data tree without editing code, e.g. on an external drive:

```bash
export TC_DATA_ROOT="/Volumes/MyDrive/nisar_swe/data"
```

## Data availability

The raw inputs are **not** included in this repository (they are large and
distributed by their original providers). One small derived input — the 932 KB
WorldCover water mask used for the ENL calibration — *is* shipped, in
`code/cache/`.

- **ASO** snow depth / SWE / density rasters, Tuolumne basin — from the Airborne
  Snow Observatory, Inc. / NSIDC. Drop the per-flight archives, **as downloaded
  and still zipped** (do not unzip or rename — `prep_aso.py` extracts the layers
  it needs itself), into:

  ```
  DATA_ROOT/sub_pixel_variability/raw/aso/
      ASO_Tuolumne_<date>_AllData_and_Reports.zip
  ```

  with `<date>` exactly matching the ASO archive name. The eight flights forming
  the four accumulation pairs are:

  ```
  2023Mar02-03   2023Mar16-17
  2024Jan29      2024Feb27-28
  2025Feb08-09   2025Feb25
  2026Jan31-Feb01 2026Feb27-28
  ```

  Running `ASO/run_aso_pipeline.sh` then unzips and processes them into
  `DATA_ROOT/sub_pixel_variability/processed/aso/`.
- **NISAR GUNWs** — the two delivered standard GUNWs (descending path 042,
  2025-12-07→19 and 2025-12-19→31) are downloaded from the Alaska Satellite
  Facility (ASF) DAAC by `code/nisar/download/fetch_paper_gunws.py`. It uses
  [`nisar_pytools`](https://github.com/zmhoppinen/nisar_pytools) and requires
  Earthdata Login credentials in `~/.netrc`. (RSLCs are *not* needed — the paper
  uses the delivered GUNWs directly, no SAS run; `download_rslcs.py` is kept only
  to document how the exploratory RSLCs were obtained.)

## Environment

```bash
conda env create -f environment.yml
conda activate nisar_pytools
# the analysis helper used by the download + NISAR scripts:
pip install git+https://github.com/zmhoppinen/nisar_pytools
```

## Reproducing the figures and statistics

With the environment active and `~/.netrc` set up, from `code/`:

```bash
cd code

# 1. ASO tier: raw flight zips -> per-pair ΔSWE -> per-window M
./ASO/run_aso_pipeline.sh

# 2. NISAR tier: fetch the two GUNWs from ASF, then calibrate + debias coherence
python nisar/download/fetch_paper_gunws.py
bash nisar/generate_nisar_products.sh

# 3. render the manuscript figures and the reported statistics
./make_all_figures.sh
./run_stats.sh
```

By default this reads/writes inside the repo (`./data`, `./figures`). To run
against data that lives elsewhere, set `TC_DATA_ROOT` (and optionally
`TC_FIG_ROOT`) first.

This pipeline has been verified to regenerate every figure in `figures/`
pixel-for-pixel, including the NISAR coherence figures rebuilt from GUNWs
freshly downloaded from ASF.

## Manuscript

`manuscript/` is a self-contained Copernicus submission package. Build with:

```bash
cd manuscript
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## License

MIT (see `LICENSE`). The manuscript text and figures under `manuscript/` are the
authors' work; if you intend to reuse those specifically, prefer citing the
paper.
