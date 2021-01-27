
import os

import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()

PREFIX = "/run/media/vap30/Hathor/work/plotting_data/Vs30/"
YCA_MAP = os.path.join(PREFIX, "IwahashiPike_NZ_100m_16.tif")


def model_prior():
    """
    names of levels
    01 - Well dissected alpine summits, mountains, etc.
    02 - Large volcano, high block plateaus, etc.
    03 - Well dissected, low mountains, etc.
    04 - Volcanic fan, foot slope of high block plateaus, etc.
    05 - Dissected plateaus, etc.
    06 - Basalt lava plain, glaciated plateau, etc.
    07 - Moderately eroded mountains, lava flow, etc.
    08 - Desert alluvial slope, volcanic fan, etc.
    09 - Well eroded plain of weak rocks, etc.
    10 - Valley, till plain, etc.
    11 - Eroded plain of weak rocks, etc.
    12 - Desert plain, delta plain, etc.
    13 - Incised terrace, etc.
    14 - Eroded alluvial fan, till plain, etc.
    15 - Dune, incised terrace, etc.
    16 - Fluvial plain, alluvial fan, low-lying flat plains, etc.
    """
    # position 13 (idx 12) was a guess - Kevin
    # fmt: off
    return np.array([[519, 0.3521],
                     [393, 0.4161],
                     [547, 0.4695],
                     [459, 0.3540],
                     [402, 0.3136],
                     [345, 0.2800],
                     [388, 0.4161],
                     [374, 0.3249],
                     [497, 0.3516],
                     [349, 0.2800],
                     [328, 0.2736],
                     [297, 0.2931],
                     [500, 0.5],
                     [209, 0.1749],
                     [363, 0.2800],
                     [246, 0.2206]], dtype=np.float32)
    # fmt: on


def model_posterior():
    """
    Update prior model based on observations.
    """
    prior = model_prior
    return []


def model_posterior_paper():
    """
    Posterior model from the paper.
    """
    # fmt: off
    return np.array([[519, 0.5],
                     [393, 0.5],
                     [547, 0.5],
                     [459, 0.5],
                     [323.504047697085, 0.407430975749267],
                     [300.639538792274, 0.306594194335118],
                     [535.786417756242, 0.380788655293195],
                     [514.970575501527, 0.380788655293195],
                     [284.470950299338, 0.360555127546399],
                     [317.368444092771, 0.33166247903554],
                     [266.870166096091, 0.4],
                     [297, 0.5],
                     [216.62368022053, 0.254950975679639],
                     [241.80869962047, 0.307482445914323],
                     [198.64481157494, 0.205256370217328],
                     [202.125866002814, 0.206820130727133]], dtype=np.float32)
    # fmt: on


def gidx(points):
    """
    Returns the category ID index for given locations. 255 for NaN.
    points: 2D numpy array of NZTM coords
    """
    raster = gdal.Open(YCA_MAP, gdal.GA_ReadOnly)
    transform = raster.GetGeoTransform()
    band = raster.GetRasterBand(1)
    # 255, convert it to raster data type dynamically?
    nodata = int(band.GetNoDataValue())

    # np.round would give duplicate pairs in some (default) grids
    # just floor because coords are left edges of pixels
    # origin 50 metres off, spacing 100
    xpos = np.floor((points[:, 0] - transform[0]) / transform[1]).astype(np.int32)
    ypos = np.floor((points[:, 1] - transform[3]) / transform[5]).astype(np.int32)
    # assume not given values out of range for this raster
    # this raster gives uint8 values, 255 for nan
    values = band.ReadAsArray()[ypos, xpos]
    # minus 1 because ids start at 1
    values = np.where(values == nodata, nodata, values - 1)

    return values


def gidx2val(model, gidx):
    """
    
    """
    vals = np.empty((len(gidx), 2), dtype=np.float32)
    vals[...] = np.nan

    valid_idx = gidx != 255
    vals[valid_idx] = model[gidx[valid_idx]]

    return vals


def save(vals, x0, y0, nx, ny, xd, yd, filename):
    driver = gdal.GetDriverByName("GTiff")
    # https://gdal.org/drivers/raster/gtiff.html
    # TILED=YES much smaller (entire nan blocks), slower with eg: QGIS
    # COMPRESS=DEFLATE smaller, =LZW larger
    gfile = driver.Create(filename, xsize=nx, ysize=ny, bands=1,
                          eType=gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
    gfile.SetGeoTransform([x0, xd, 0, y0, 0, yd])
    # projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2193)
    gfile.SetProjection(srs.ExportToWkt())
    # data
    gfile.GetRasterBand(1).WriteArray(vals.reshape(ny, nx))
    # close file
    gfile = None
    

# grid setup
x0 = 1000050
xn = 2126350
y0 = 4700050
yn = 6338350
xd = 100
yd = 100
nx = round((xn - x0) / xd + 1)
ny = round((yn - y0) / yd + 1)

# run
points = np.vstack(np.mgrid[x0:xn + 1:xd,y0:yn + 1:yd].T)
gids = gidx(points)
model = model_posterior_paper()
vals = gidx2val(model, gids)
save(vals[:, 0], x0, y0, nx, ny, xd, yd, "terrain.tif")
