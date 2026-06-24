# Snow accumulation variability limits InSAR SWE retrieval

Companion code and manuscript for the study of how sub-resolution snow
accumulation (ΔSWE) variability biases and decorrelates L-band InSAR
snow-water-equivalent retrievals (NISAR-relevant), evaluated over the
Tuolumne basin using Airborne Snow Observatory (ASO) lidar.

## Repository layout

```
code/         analysis and figure-generation scripts
  ASO/          ASO lidar -> per-pair ΔSWE and the per-window factor M
  nisar/        NISAR GUNW prep, ENL calibration, coherence debiasing
  stats/        statistics reported in the paper
  visualizations/  the figure scripts (plus shared plotting_constants.py)
  make_all_figures.sh   regenerate the manuscript figures
  run_stats.sh          regenerate the reported statistics
  STATS_MAP.md          maps each reported number to the script that makes it
manuscript/   self-contained Copernicus LaTeX source (template.tex) + figures
figures/      generated output figures (so results are viewable without the data)
environment.yml   conda environment used for the analysis
```

## Data availability

The raw inputs are **not** included in this repository (they are large and
distributed by their original providers):

- **ASO** snow depth / SWE / density rasters, Tuolumne basin — Airborne Snow
  Observatory, Inc. / NSIDC.
- **NISAR RSLCs and GUNWs** — downloaded from the Alaska Satellite Facility
  (ASF) DAAC by `code/nisar/download/` (`download_rslcs.py`,
  `download_gunws.py`), and GUNWs additionally generated from RSLCs via
  `code/nisar/generate_nisar_products.sh`. These use
  [`nisar_pytools`](https://github.com/zmhoppinen/nisar_pytools) and require
  Earthdata Login credentials in `~/.netrc`.

The scripts currently use **absolute paths** to the author's local data tree
(e.g. `/Users/zmhoppinen/Documents/nisar_swe/data/...`) and write figures back
to a local `figures/` tree. They are published as a transparent record of the
analysis rather than a turnkey pipeline; reproducing the results requires
pointing those paths at a local copy of the data.

## Environment

```bash
conda env create -f environment.yml
conda activate nisar_pytools
```

## Reproducing the figures and statistics

With the environment active and data paths updated:

```bash
cd code
./make_all_figures.sh    # manuscript figures
./run_stats.sh           # reported statistics
```

## Manuscript

`manuscript/` is a self-contained Copernicus submission package. Build with:

```bash
cd manuscript
pdflatex template.tex && bibtex template && pdflatex template.tex && pdflatex template.tex
```

## License

No license is set yet. Without one, the default is "all rights reserved."
Add a `LICENSE` file (e.g. MIT for code, CC-BY 4.0 for the manuscript/figures)
before relying on others being able to reuse this.
