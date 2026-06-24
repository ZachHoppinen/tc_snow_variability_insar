"""interval_accumulation.py -- per-interval snow accumulation over the two NISAR
12-day windows, to justify that P23 is the larger-accumulation pair.

NISAR Dec-2025 triplet (Tuolumne, descending track 069 / frame 042):
    P12 : 2025-12-07 -> 2025-12-19
    P23 : 2025-12-19 -> 2025-12-31

ASO has no coincident Dec-2025 flight, so the per-interval accumulation reference
is daily SWE at the Tuolumne-basin CDEC snow pillows (sensor 82, snow water
content). For each station we difference SWE between the interval endpoints:
    dSWE_interval = SWE(end) - SWE(start)        (positive = net accumulation)
then average across stations. If mean(dSWE_P23) > mean(dSWE_P12), the snow signal
driving the InSAR phase is larger in P23.

Source: CDEC CSV web service (public). Network required.
"""

from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# Tuolumne-basin CDEC snow pillows (edit here to add/drop stations)
# ============================================================================
STATIONS = {
    "DAN": "Dana Meadows",      # ~2987 m
    "TUM": "Tuolumne Meadows",  # ~2622 m
    "SLI": "Slide Canyon",      # ~2804 m
    "GIN": "Gin Flat",          # ~2149 m
}
SENSOR = 82                                   # daily snow water content (inches)
IN_TO_MM = 25.4
INTERVALS = {                                 # the NISAR pairs (+ the 12-day prior)
    "P01": ("2025-11-25", "2025-12-07"),      # the pair before P12 (reference check)
    "P12": ("2025-12-07", "2025-12-19"),
    "P23": ("2025-12-19", "2025-12-31"),
    "P34": ("2025-12-31", "2026-01-12"),      # the pair after P23 (2nd storm check)
}
FETCH_START, FETCH_END = "2025-11-21", "2026-01-16"   # pad around the windows

STATS_DIR = Path(__file__).resolve().parent
FIG_DIR = Path("/Users/zmhoppinen/Documents/nisar_swe/figures/"
               "ASO_sweramp_unwrapping_coherence/v4")


def fetch_swe(station: str) -> pd.Series:
    """Daily SWE (mm) time series for one CDEC station over the padded window."""
    q = urlencode({"Stations": station, "SensorNums": SENSOR, "dur_code": "D",
                   "Start": FETCH_START, "End": FETCH_END})
    url = f"https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?{q}"
    df = pd.read_csv(url)                       # pandas pulls the CSV itself
    df["DATE"] = pd.to_datetime(df["DATE TIME"], format="%Y%m%d %H%M", errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    s = df.dropna(subset=["DATE"]).set_index("DATE")["VALUE"].sort_index()
    return s * IN_TO_MM                         # inches -> mm


def swe_on(series: pd.Series, date: str) -> float:
    """SWE at a date, nearest valid daily value within +/-2 days (else NaN)."""
    s = series.dropna()
    if s.empty:
        return np.nan
    hit = s.reindex([pd.Timestamp(date)], method="nearest",
                    tolerance=pd.Timedelta("2D"))
    return float(hit.iloc[0])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- step 1: fetch SWE for each station ---------------------------------
    series = {}
    for sid, name in STATIONS.items():
        try:
            s = fetch_swe(sid)
            series[sid] = s
            print(f"step 1  {sid} ({name}): {s.dropna().size} daily SWE obs, "
                  f"range {np.nanmin(s.values):.0f}-{np.nanmax(s.values):.0f} mm")
        except Exception as e:
            print(f"step 1  {sid} ({name}): FETCH FAILED -> {e}")

    # --- step 2: per-station dSWE over each interval ------------------------
    rows = []
    for sid in series:
        row = {"station": sid, "name": STATIONS[sid]}
        for pair, (d0, d1) in INTERVALS.items():
            a, b = swe_on(series[sid], d0), swe_on(series[sid], d1)
            row[f"{pair}_start_mm"] = a
            row[f"{pair}_end_mm"] = b
            row[f"{pair}_dSWE_mm"] = b - a
        rows.append(row)
    table = pd.DataFrame(rows).set_index("station")
    print("\nstep 2  per-station dSWE (mm):")
    print(table[[f"{p}_dSWE_mm" for p in INTERVALS]].round(1).to_string())

    # --- step 3: average across stations + justification --------------------
    means = {p: table[f"{p}_dSWE_mm"].mean() for p in INTERVALS}
    print("\nstep 3  basin-mean dSWE (mm):")
    for p in INTERVALS:
        d0, d1 = INTERVALS[p]
        print(f"          {p} ({d0} -> {d1}): {means[p]:+.1f} mm "
              f"(n={table[f'{p}_dSWE_mm'].notna().sum()} stations)")
    verdict = ("P23 > P12 -> P23 is the larger-accumulation pair"
               if means["P23"] > means["P12"] else
               "P23 <= P12 -> P23 is NOT the larger-accumulation pair")
    print(f"        => {verdict}")

    out_csv = STATS_DIR / "interval_accumulation.csv"
    table.round(1).to_csv(out_csv)
    print(f"        wrote {out_csv.name}")

    # --- step 4: figure -- SWE traces + per-interval mean dSWE --------------
    shade = {"P01": "tab:gray", "P12": "tab:blue", "P23": "tab:green",
             "P34": "tab:red"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for sid in series:
        s = series[sid].dropna()
        ax[0].plot(s.index, s.values, lw=1.6, label=f"{sid} {STATIONS[sid]}")
    for p, (d0, d1) in INTERVALS.items():       # shade the two windows
        ax[0].axvspan(pd.Timestamp(d0), pd.Timestamp(d1), color=shade[p],
                      alpha=0.12, label=f"{p} window")
    ax[0].set_ylabel("SWE (mm)")
    ax[0].set_title("Tuolumne CDEC pillows, Dec 2025")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].tick_params(axis="x", rotation=30)

    x = np.arange(len(INTERVALS))
    ax[1].bar(x, [means[p] for p in INTERVALS],
              color=[shade[p] for p in INTERVALS], alpha=0.7, width=0.6)
    for j, sid in enumerate(series):            # overlay per-station points
        ax[1].scatter(x, [table.loc[sid, f"{p}_dSWE_mm"] for p in INTERVALS],
                      color="0.25", s=30, zorder=3,
                      label="stations" if j == 0 else None)
    ax[1].axhline(0, color="0.4", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(
        [f"{p}\n{INTERVALS[p][0][5:]}->{INTERVALS[p][1][5:]}" for p in INTERVALS])
    ax[1].set_ylabel(r"$\Delta$SWE over interval (mm)")
    ax[1].set_title("per-interval accumulation (bar = basin mean)")
    ax[1].legend(fontsize=9)

    out = FIG_DIR / "interval_accumulation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
