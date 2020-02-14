
import numpy as np
import pandas as pd
from pyproj import Proj, transform


wgs84 = Proj(init="epsg:4326")
nzmg = Proj(init="epsg:27200")
#nztm = Proj(init="epsg:2193")


def downsample_mcg(df):
    """Resample McGann points on 1km grid."""
    # DOESNT SEEM TO DO ANYTHING

    # meters
    # too far just means the point is further away than the centre `sqrt(grid_res**2 * 2) / 2` (only possible around edges)
    too_far = 707.2
    grid_res = 1000

    # Make the grid. Use raster() and then convert.
    # Not the most direct but it works.
    ggg = raster(crs = proj4string(inputSPDF),
                ext = extent(bbox(inputSPDF)),
                resolution = grid_res)
    gg = as(ggg, 'SpatialGrid')
    g = as(gg, 'SpatialPoints')

    # Note that with 1000 meter spacing, a grid overlay on McGann points
    # ends up as 21 x 43 km, with 22x44 points = 968 points:
    # > points2grid(gg)
    # s1      s2
    # cellcentre.offset 2468048 5725866
    # cellsize             1000    1000
    # cells.dim              22      44

    # > dim(coordinates(g))
    # [1] 968   2

    # converting to "ppp"...
    grd = ppp(x = coordinates(g)[,1],
        y = coordinates(g)[,2],
        window = owin(xrange=bbox(g)[1,], yrange = bbox(g)[2,]))

    inpSPDFppp = ppp(x = coordinates(inputSPDF)[,1],
        y = coordinates(inputSPDF)[,2],
        window = owin(xrange=bbox(inputSPDF)[1,], yrange=bbox(inputSPDF)[2,]))

    # Finally, running nncross
    distz = nncross(grd, inpSPDFppp)

    # I want to take only the points in McGann original data that are less than "tooFar"
    # distance away from one of the grid points.

    # Throw out all data too far from a McGann point.
    distz <- distz[distz$dist < tooFar,]

    # Remove all other points from input dataframe
    outp <- inputSPDF[distz$which,] 

    return(outp)


def load_vs(downsample_mcgann=True):
    """
    As of 2017-11 this comprises 3 sources
    McGann Vs30 map (McGann, submitted, 2016 SDEE "Development of regional Vs30 Model and typical profiles... Christchurch CPT correlation")
    Kaiser et al.
    Internal communication with Wotherspoon: Characterised Vs30 Canterbury_June2017_KFed.csv
    """

    # load each Vs data source.
    mcgann = load_mcgann_vs(downsample_mcgann)
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
