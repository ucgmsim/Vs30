from math import sqrt

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import distance_matrix

DATA_CPT = "data/cptvs30.ssv"
DATA_KAISERETAL = "data/20170817_vs_allNZ_duplicatesCulled.ll"
DATA_MCGANN = "data/McGann_cptVs30data.csv"
DATA_WOTHERSPOON = "data/Characterised Vs30 Canterbury_June2017_KFed.csv"

wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)
nzmg2nztm = Transformer.from_crs(27200, 2193, always_xy=True)


def downsample_mcg(df, res=1000):
    """
    Resample McGann points on 1km grid.
    res: grid resolution (m)
    """

    max_dist = sqrt(res ** 2 * 2) / 2
    x = df["easting"].values
    y = df["northing"].values

    # trying to copy R logic - extents
    # works for this dataset
    xmin = min(x)
    nx = round((max(x) - xmin) / res)
    if ((max(x) - xmin) / res) % 1 < 0.5:
        # this is run
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
        # this is run
        ymax += res / 2.0
        ymin = ymax - (ny - 1) * res

    # coarse grid
    xx = np.linspace(xmin, xmax, nx)
    yy = np.linspace(ymin, ymax, ny)[::-1]
    grid = np.dstack(np.meshgrid(xx, yy)).reshape(-1, 2)

    # distances from coarse grid, nearest neighbor
    dist = distance_matrix(grid, np.dstack((x, y))[0])
    nn = np.argmin(dist, axis=1)
    # cut out if no points within search area
    nn = nn[dist[np.arange(nn.size), nn] <= max_dist]

    # remove duplicate points from downsample algorithm
    mcg = df.iloc[nn]
    return mcg[~mcg.duplicated()]


def load_vs(cpt=False, downsample_mcgann=True):
    """
    Load measured sites. Either newer CPT based or paper version using 3 sources.
    """

    if cpt:
        return load_cpt_vs()

    # load each Vs data source
    mcgann = load_mcgann_vs(downsample=downsample_mcgann)
    wotherspoon = load_wotherspoon_vs()
    kaiseretal = load_kaiseretal_vs()

    # remove Kaiser Q3 unless station name is 3 chars long
    kaiseretal = kaiseretal[(kaiseretal.q != 3) | (kaiseretal.station.str.len() == 3)]

    return pd.concat([mcgann, wotherspoon, kaiseretal], ignore_index=True)


def load_cpt_vs():
    """
    Newer collection of Vs30 from CPT data.
    """
    cpt = pd.read_csv(
            DATA_CPT,
            sep=" ",
            usecols=[0, 1, 2],
            names=["easting", "northing", "vs30"],
            skiprows=1,
            engine="c",
            dtype=np.float32,
        )
    # remove rows with no vs30 value
    cpt = cpt[~np.isnan(cpt.vs30)]

    cpt["uncertainty"] = np.float32(0.5)

    return cpt

def load_mcgann_vs(downsample=True):
    """
    McGann Vs30 map
    (McGann, submitted, 2016 SDEE
    "Development of regional Vs30 Model and typical profiles... Christchurch CPT correlation")
    """

    # downsampling was originally done on the NZMG grid
    mcgann = pd.read_csv(
        DATA_MCGANN,
        usecols=[3, 4, 7] if downsample else [5, 6, 7],
        names=["easting", "northing", "vs30"],
        skiprows=1,
        engine="c",
        dtype=np.float32,
    )
    if downsample:
        mcgann = downsample_mcg(mcgann)
        mcgann["easting"], mcgann["northing"] = nzmg2nztm.transform(
            mcgann["easting"].values, mcgann["northing"].values
        )

    mcgann["uncertainty"] = np.float32(0.2)

    return mcgann


def load_wotherspoon_vs():
    """
    Internal communication with Wotherspoon: Characterised Vs30
    """

    wotherspoon = pd.read_csv(
        DATA_WOTHERSPOON,
        sep="\t",
        usecols=[2, 3, 4],
        names=["northing", "easting", "vs30"],
        skiprows=1,
        engine="c",
        dtype=np.float32,
    )
    wotherspoon["easting"], wotherspoon["northing"] = wgs2nztm.transform(
        wotherspoon["easting"].values, wotherspoon["northing"].values
    )

    wotherspoon["uncertainty"] = np.float32(0.2)

    return wotherspoon


def load_kaiseretal_vs():
    """
    Values from Kaiser et al. (2017)
    """

    # with removed duplicate points
    kaiseretal = pd.read_csv(
        DATA_KAISERETAL,
        usecols=[0, 1, 2, 3, 4],
        names=["station", "easting", "northing", "vs30", "q"],
        skiprows=5,
        engine="c",
        dtype={"easting": np.float32, "northing": np.float32, "vs30": np.float32},
        converters={"q": lambda text: int(text.strip()[2]), "station": str.strip},
    )
    kaiseretal["easting"], kaiseretal["northing"] = wgs2nztm.transform(
        kaiseretal["easting"].values, kaiseretal["northing"].values
    )

    # [Q1, Q2 and Q3] correspond to approximate uncertainties
    # of <10%, 10-20% and >20% respectively
    # 10% : ln(1.1) ~= 0.1
    # 20% : ln(1.2) ~= 0.2
    # 50% : ln(1.5) ~= 0.5
    kaiseretal["uncertainty"] = np.where(
        kaiseretal["q"] == 3, 0.5, kaiseretal["q"] / 10
    )

    return kaiseretal
