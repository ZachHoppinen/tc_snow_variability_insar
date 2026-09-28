"""Single-look ISCE3 InSAR run on a cropped Tuolumne pair.

Builds the runconfig from ONELOOK_TEMPLATE.yaml next to this script (crossmul
1 x 1, dense offsets + rubbersheeting on, unwrap at 13 x 16) with the paths and
geocode box for the requested AOI, then launches nisar.workflows.insar in the
isce3 env. The product we want is <pair>/scratch/RIFG.h5, the single-look
interferogram in radar geometry; the geocoded product.h5 is a by-product.

Output (local, under the repo): data/processed/nisar/onelook/<aoi>/<ref>_<sec>/

Run from the repo root (any env with PyYAML):
    python code/nisar/onelook/run_onelook_pair.py aoi3 20251219 20251231
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import yaml

from onelook_config import AOIS, HALF, OUT_ROOT

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "ONELOOK_TEMPLATE.yaml"
ISCE3_PY = Path.home() / "miniforge3/envs/isce3/bin/python"


def write_runconfig(aoi: str, ref: str, sec: str) -> tuple[Path, Path]:
    crops = OUT_ROOT / aoi / "crops"
    d = OUT_ROOT / aoi / f"{ref}_{sec}"
    (d / "scratch").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(TEMPLATE.read_text())
    g = cfg["runconfig"]["groups"]
    g["input_file_group"]["reference_rslc_file"] = str(crops / f"{ref}_sub.h5")
    g["input_file_group"]["secondary_rslc_file"] = str(crops / f"{sec}_sub.h5")
    g["input_file_group"]["qa_gunw_input_file"] = str(d / "product.h5")
    g["dynamic_ancillary_file_group"]["dem_file"] = str(crops / "dem.tif")
    g["product_path_group"].update({
        "product_path": str(d), "scratch_path": str(d / "scratch"),
        "sas_output_file": str(d / "product.h5"), "qa_output_dir": str(d / "qa_insar")})
    # geocode box = the crop box, in the GUNW grid (EPSG:32610)
    cx, cy = AOIS[aoi]
    for key in ("geocode", "radar_grid_cubes"):
        g["processing"][key]["output_epsg"] = 32610
        g["processing"][key]["top_left"] = {"x_abs": cx - HALF, "y_abs": cy + HALF}
        g["processing"][key]["bottom_right"] = {"x_abs": cx + HALF, "y_abs": cy - HALF}
    cfg["runconfig"]["groups"]["logging"]["path"] = str(d / "scratch/insar.log")
    rc = d / "runconfig.yaml"
    rc.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return d, rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("aoi", choices=AOIS)
    ap.add_argument("ref"); ap.add_argument("sec")
    a = ap.parse_args()
    d, rc = write_runconfig(a.aoi, a.ref, a.sec)
    if (d / "scratch/RIFG.h5").exists():
        print(f"{a.ref}_{a.sec}: have RIFG.h5, skip"); return
    log = d / "run.log"
    print(f"{a.ref}_{a.sec}: running, log {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run([str(ISCE3_PY), "-m", "nisar.workflows.insar", str(rc)],
                           stdout=fh, stderr=subprocess.STDOUT)
    print(f"{a.ref}_{a.sec}: exit {r.returncode} after {(time.time() - t0) / 60:.1f} min; "
          f"RIFG present: {(d / 'scratch/RIFG.h5').exists()}", flush=True)


if __name__ == "__main__":
    main()
