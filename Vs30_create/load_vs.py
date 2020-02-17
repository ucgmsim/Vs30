
from math import sqrt

import numpy as np
import pandas as pd
from pyproj import Proj, transform
from scipy.spatial import distance_matrix


wgs84 = Proj(init="epsg:4326")
nzmg = Proj(init="epsg:27200")
#nztm = Proj(init="epsg:2193")


def downsample_mcg(df, res=1000):
    """
    Resample McGann points on 1km grid.
    res: grid resolution (m)
    """

    max_dist = sqrt(res ** 2 * 2) / 2
    x = df["Easting"].values
    y = df["Northing"].values

    # trying to copy R logic - extents
    xmin = round(min(x))
    nx = round((max(x) - xmin) / res)
    xmax = nx * res + xmin
    ymax = round(max(y))
    ny = round((ymax - min(y)) / res)
    ymin = ymax - ny * res

    # coarse grid
    xx = np.linspace(xmin, xmax, nx)
    yy = np.linspace(ymin, ymax, ny)
    grid = np.dstack(np.meshgrid(xx, yy)).reshape(-1, 2)

    # distances from coarse grid, nearest neighbor
    dist = distance_matrix(grid, np.dstack((x, y))[0])
    nn = np.argmin(dist, axis=1)
    # cut out if no points within search area
    nn = nn[dist[np.arange(nn.size), nn] <= max_dist]
    nn.sort()

    return df.iloc[nn]


def load_vs(downsample_mcgann=True):
    """
    As of 2017-11 this comprises 3 sources
    McGann Vs30 map (McGann, submitted, 2016 SDEE "Development of regional Vs30 Model and typical profiles... Christchurch CPT correlation")
    Kaiser et al.
    Internal communication with Wotherspoon: Characterised Vs30 Canterbury_June2017_KFed.csv
    """

    # load each Vs data source.
    mcgann = load_mcgann_vs(downsample=downsample_mcgann)
    wotherspoon = load_wotherspoon_vs()
    kaiseretal = load_kaiseretal_vs()

    vs = pd.concat([mcgann, wotherspoon, kaiseretal], ignore_index=True)

    return vs


def load_mcgann_vs(downsample=True):
    # note this uses nzmg cols, wotherspoon starts with lon/lat and goes to nzmg

    PATH = "McGann_cptVs30data.csv"
    mcgann = pd.read_csv(
        PATH,
        usecols=[3, 4, 7],
        names=["Easting", "Northing", "Vs30"],
        skiprows=1,
        engine="c",
        dtype=np.float32,
    )

    mcgann["Uncertainty"] = np.float32(0.2)

    # apply downsampling criterion to McGann CPT-based data
    if downsample:
        return downsample_mcg(mcgann)
    return mcgann


def load_wotherspoon_vs():
    """Liam Wotherspoon's spreadsheet for the Canterbury region."""

    PATH = "Characterised Vs30 Canterbury_June2017_KFed.csv"
    wotherspoon = pd.read_csv(
        PATH,
        sep="\t",
        usecols=[2, 3, 4],
        names=["Northing", "Easting", "Vs30"],
        skiprows=1,
        engine="c",
        dtype=np.float32,
    )
    easts, norths = transform(
        wgs84, nzmg, wotherspoon["Easting"].values, wotherspoon["Northing"].values
    )
    wotherspoon["Easting"] = easts
    wotherspoon["Northing"] = norths

    wotherspoon["Uncertainty"] = np.float32(0.2)

    return wotherspoon


def load_kaiseretal_vs():
    """Values from Kaiser et al. (2017)"""

    # with removed duplicate points
    PATH = "20170817_vs_allNZ_duplicatesCulled.ll"
    kaiseretal = pd.read_csv(
        PATH,
        usecols=[1, 2, 3, 4],
        names=["Easting", "Northing", "Vs30", "Q"],
        skiprows=5,
        engine="c",
        dtype={"Easting": np.float32, "Northing": np.float32, "Vs30": np.float32},
        converters={"Q": lambda text: int(text.strip()[2])},
    )
    easts, norths = transform(
        wgs84, nzmg, kaiseretal["Easting"].values, kaiseretal["Northing"].values
    )
    kaiseretal["Easting"] = easts
    kaiseretal["Northing"] = norths

    kaiseretal["Uncertainty"] = np.float32(np.nan)
    # Kaiser et al. states:
    #   "[Q1, Q2 and Q3] correspond to approximate uncertainties
    #    of <10%, 10-20% and >20% respectively."
    #
    #
    # I choose the following values:
    #
    #     10%    :   ln(1.1) ~= 0.1
    #     20%    :   ln(1.2) ~= 0.2
    #     50%    :   ln(1.5) ~= 0.5
    #
    kaiseretal["Uncertainty"][kaiseretal["Q"] == 1] = 0.1
    kaiseretal["Uncertainty"][kaiseretal["Q"] == 2] = 0.2
    kaiseretal["Uncertainty"][kaiseretal["Q"] == 3] = 0.5

    del kaiseretal["Q"]

    return kaiseretal
