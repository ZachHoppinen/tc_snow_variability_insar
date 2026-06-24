"""Repository-relative data and figure roots.

Every script derives its input/output locations from these two roots instead
of hardcoding absolute paths, so the repo is portable. Both default to
directories inside the repo and can be redirected with environment variables:

    TC_DATA_ROOT   where the input data lives   (default: <repo>/data)
    TC_FIG_ROOT    where figures are written     (default: <repo>/figures)

For example, to reproduce against a copy of the data on an external drive
without moving it into the repo:

    export TC_DATA_ROOT="/Volumes/MyDrive/nisar_swe/data"
"""

import os
from pathlib import Path

# code/tc_paths.py  ->  parent is code/, parent.parent is the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("TC_DATA_ROOT", REPO_ROOT / "data"))
FIG_ROOT = Path(os.environ.get("TC_FIG_ROOT", REPO_ROOT / "figures"))
