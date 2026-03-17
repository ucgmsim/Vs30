"""
Create the clean combined observation CSV for the Jaehwi version of the Vs30 model.

Jaehwi's input data consists of three files:
  1. McGann_cptVs30data.csv — same McGann CPT data as the legacy codebase
  2. Measured_Vs30data.csv — newly compiled surface-wave Vs30 measurements
  3. Geonet Metadata Summary_v1.4_NOVs30map.csv — updated GeoNet station metadata
     (superset of the legacy Kaiser et al. data, with per-station Sigmaln_Vs30)

The legacy codebase applied two filters to its observation data:
  - McGann data was downsampled on a 1km NZMG grid
  - Kaiser et al. Q3 stations were removed unless the station name was exactly
    3 characters long (broadband seismometers on rock)
These same filters are applied here: McGann is downsampled, and Geonet Q3
stations are removed unless the station name is 3 characters.
Measured_Vs30data is all Q2 so no Q3 filtering applies.

Uncertainty assignment:
  - McGann: constant 0.2 (legacy assumed value)
  - Measured_Vs30data: uses per-entry σln(Vs30) where available, otherwise
    Q-based approximation (Q2=0.2) as in the legacy code
  - Geonet metadata: uses per-entry Sigmaln_Vs30 (actual values provided)
"""

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import distance_matrix

DATA_DIR = Path(__file__).parent / "raw_data"
REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "vs30" / "resources" / "observations"

wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)
nzmg2nztm = Transformer.from_crs(27200, 2193, always_xy=True)


def downsample_mcgann(sites_df, res=1000):
    """
    Resample McGann points on 1km grid (reproduces legacy R-derived logic).
    """
    max_dist = sqrt(res**2 * 2) / 2
    x = sites_df["easting"].values
    y = sites_df["northing"].values

    xmin = min(x)
    nx = round((max(x) - xmin) / res)
    if ((max(x) - xmin) / res) % 1 < 0.5:
        xmin += res / 2.0
        xmax = (nx - 1) * res + xmin
    else:
        xmax = nx * res + xmin
    ymax = max(y) - res
    ny = round((ymax - min(y)) / res) + 1
    if ((ymax - min(x)) / res) % 1 < 0.5:
        ymax -= res / 2.0
        ymin = ymax - (ny - 2) * res
    else:
        ymax += res / 2.0
        ymin = ymax - (ny - 1) * res

    grid_x = np.linspace(xmin, xmax, nx)
    grid_y = np.linspace(ymin, ymax, ny)[::-1]
    grid = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

    dist = distance_matrix(grid, np.dstack((x, y))[0])
    nn = np.argmin(dist, axis=1)
    nn = nn[dist[np.arange(nn.size), nn] <= max_dist]

    mcg = sites_df.iloc[nn]
    return mcg[~mcg.duplicated()]


def process_mcgann():
    """Load and downsample McGann CPT data (same as legacy)."""
    mcgann = pd.read_csv(
        DATA_DIR / "McGann_cptVs30data.csv",
        usecols=[3, 4, 7],
        names=["easting", "northing", "vs30"],
        skiprows=1,
        engine="c",
        dtype=np.float32,
    )
    mcgann = downsample_mcgann(mcgann)
    mcgann["easting"], mcgann["northing"] = nzmg2nztm.transform(
        mcgann["easting"].values, mcgann["northing"].values
    )
    mcgann["uncertainty"] = np.float32(0.2)
    mcgann["source"] = "mcgann"
    mcgann["station"] = ""
    mcgann["q"] = ""
    return mcgann[["easting", "northing", "vs30", "uncertainty", "source", "station", "q"]]


def process_measured():
    """
    Load Measured_Vs30data.csv (newly introduced by Jaehwi).
    All entries are Q2. Uses per-entry σln(Vs30) where available, otherwise 0.2.
    """
    measured = pd.read_csv(DATA_DIR / "Measured_Vs30data.csv")
    easting, northing = wgs2nztm.transform(
        measured["Longitude"].values, measured["Latitude"].values
    )
    # Use per-entry sigma if available, otherwise Q-based default (all Q2 -> 0.2)
    uncertainty = measured["σln(VS30)"].values.astype(float)
    uncertainty = np.where(np.isnan(uncertainty), 0.2, uncertainty)

    result = pd.DataFrame({
        "easting": easting,
        "northing": northing,
        "vs30": measured["Vs30"].values,
        "uncertainty": uncertainty,
        "source": "measured_vs30",
        "station": measured["Station"].values,
        "q": measured["q"].values,
    })
    return result


def process_geonet():
    """
    Load Geonet Metadata Summary.
    Apply legacy Q3 filter: remove Q3 stations unless station name is exactly
    3 characters (broadband seismometers on rock).
    Uses per-entry Sigmaln_Vs30 values.
    """
    geonet = pd.read_csv(
        DATA_DIR / "Geonet Metadata Summary_v1.4_NOVs30map.csv"
    )
    easting, northing = wgs2nztm.transform(
        geonet["Longitude"].values, geonet["Latitude"].values
    )
    q_numeric = geonet["Q_Vs30"].str[1].astype(int)

    result = pd.DataFrame({
        "easting": easting,
        "northing": northing,
        "vs30": geonet["Vs30"].values,
        "uncertainty": geonet["Sigmaln_Vs30"].values,
        "source": "geonet_metadata",
        "station": geonet["Station"].values,
        "q": q_numeric.values,
    })

    # Legacy Q3 filter: remove Q3 unless station name is 3 characters
    is_q3 = result["q"] == 3
    is_3char = result["station"].str.len() == 3
    remove = is_q3 & ~is_3char

    n_removed = remove.sum()
    print(f"Geonet: removing {n_removed} Q3 stations (non-3-char name)")
    return result[~remove].reset_index(drop=True)


def main():
    mcgann = process_mcgann()
    print(f"McGann: {len(mcgann)} entries (after 1km downsampling)")

    measured = process_measured()
    print(f"Measured_Vs30data: {len(measured)} entries")

    geonet = process_geonet()
    print(f"Geonet metadata: {len(geonet)} entries (after Q3 filtering)")

    combined = pd.concat([mcgann, measured, geonet], ignore_index=True)
    print(f"\nTotal combined: {len(combined)} entries")
    print(f"Sources: {combined['source'].value_counts().to_dict()}")

    output_path = OUTPUT_DIR / "jaehwi_v1p0_independent_observations.csv"
    comment_header = """\
# Uncertainty values are ln-scale standard deviations of Vs30.
# To convert to approximate percentage uncertainty in linear Vs30: (e^uncertainty - 1) * 100.
# For example, (e^0.1 - 1)*100 ~= 10%, (e^0.2 - 1)*100 ~= 22%, (e^0.5 - 1)*100 ~= 65%.
#
# Uncertainty values per source:
#   mcgann:         0.2 (constant, ~20%, legacy assumed value)
#   measured_vs30:  per-entry σln(Vs30) where available, otherwise 0.2 (~20%) for Q2
#   geonet_metadata: per-entry Sigmaln_Vs30 from GeoNet station metadata
#
# Filtering applied:
#   mcgann: downsampled on 1km NZMG grid (legacy behavior)
#   geonet_metadata: Q3 stations removed unless station name is 3 characters (legacy behavior)
#   measured_vs30: no filtering (all entries are Q2)
#
# Generated by dev/observations/jaehwi_v1p0/create_jaehwi_observations.py
"""
    with open(output_path, "w") as f:
        f.write(comment_header)
        combined.to_csv(f, index=False)
    print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()
