"""Shared settings for the single-look Tuolumne crop-and-run scripts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tc_paths import REPO_ROOT  # noqa: E402

# full-frame RSLCs (descending track 042, frame 069) and the frame DEM (WGS84)
RSLC_DIR = Path("/Volumes/AIALIK/2026/aso_tuolome_processing/data/nisar/rslc/tuolome")
DEM_SRC = Path("/Volumes/AIALIK/2026/aso_tuolome_processing/data/nisar/gunw/"
               "closure_triplet_tuolome/20251219_20251231/dem.tif")
# local output so the isce3 scratch traffic stays off the external drive
OUT_ROOT = REPO_ROOT / "data/processed/nisar/onelook"

# AOI centers (EPSG:32610, the GUNW grid) from scratch/find_tuolumne_aois.py;
# the 1 km AOI sits at the center of a HALF-padded crop box
AOIS = {"aoi3": (820850.0, 4202910.0),     # first test block (below treeline)
        # blocks with mean elevation >= 3200 m from the same search
        "aoi1": (818850.0, 4218910.0),
        "aoi2": (822350.0, 4207410.0),
        "aoi6": (820850.0, 4210410.0),
        # blocks from the 2600 m floor search: north-central plateau (#3) and
        # the south rim of the Tuolumne canyon (#5)
        "plateau": (800850.0, 4220410.0),
        "canyonrim": (803850.0, 4202410.0),
        # blocks from the coherence-squared search (2600 m floor), numbered as
        # in that search; 3, 4 and 7 there are aoi1, aoi6 and canyonrim
        "cs0": (822350.0, 4213910.0),
        "cs1": (822350.0, 4206910.0),
        "cs2": (821350.0, 4202910.0),
        "cs5": (828850.0, 4201410.0),
        "cs6": (815850.0, 4219910.0)}
HALF = 3000.0        # m, half side of the crop box
MARGIN = 256         # extra single-look samples on every side of the window
