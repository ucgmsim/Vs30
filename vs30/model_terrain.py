import os
from subprocess import call

import numpy as np
from osgeo import gdal, gdalconst, osr

gdal.UseExceptions()

IWAHASHI_PIKE = "IwahashiPike.tif"
IP_NODATA = 255
TERRAIN_NODATA = -32767


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


def mid(points):
    """
    Returns the category ID index for given locations.
    points: 2D numpy array of NZTM coords
    """
    raster = gdal.Open(IWAHASHI_PIKE, gdal.GA_ReadOnly)
    transform = raster.GetGeoTransform()
    band = raster.GetRasterBand(1)
    nodata = int(band.GetNoDataValue())

    xpos = np.floor((points[:, 0] - transform[0]) / transform[1]).astype(np.int32)
    ypos = np.floor((points[:, 1] - transform[3]) / transform[5]).astype(np.int32)
    # TODO: assuming not given values out of range for this raster
    values = band.ReadAsArray()[ypos, xpos]
    # minus 1 because ids start at 1
    values = np.where(values == nodata, np.nan, values - 1)

    return values


def mid_map(args):
    """
    Calculate id at map points by resampling / resizing origin id map.
    """
    path = os.path.join(args.wd, "tid.tif")
    gdal.Warp(
        path,
        os.path.join(args.mapdata, IWAHASHI_PIKE),
        creationOptions=["COMPRESS=DEFLATE"],
        outputBounds=[args.xmin, args.ymin, args.xmax, args.ymax],
        xRes=args.xd,
        yRes=args.yd,
        resampleAlg=gdalconst.GRIORA_NearestNeighbour,
    )
    return path


def mid2val(mid, model):
    """
    Convert IDs returned by the mid function into model values.
    """
    vals = np.empty((len(gidx), 2), dtype=np.float32)
    vals[...] = np.nan

    valid_idx = np.invert(np.isnan(gid))
    vals[valid_idx] = model[gid[valid_idx]]

    return vals


def model_map(args, model):
    path = os.path.join(args.wd, "terrain.tif")
    # terrain IDs for given map spec
    tid_tif = mid_map(args)

    # model version for indexing (index 0 for NODATA)
    vs30 = np.append(TERRAIN_NODATA, model[:, 0]).astype(np.float32)
    stdv = np.append(TERRAIN_NODATA, model[:, 1]).astype(np.float32)
    # string array
    vs30 = ",".join([f"{x:.8f}" for x in vs30])
    stdv = ",".join([f"{x:.8f}" for x in stdv])
    # string expression to pass into calc
    vs30 = f"numpy.array([{vs30}], dtype=numpy.float32)[numpy.where(A == {IP_NODATA}, 0, A)]"
    stdv = f"numpy.array([{stdv}], dtype=numpy.float32)[numpy.where(A == {IP_NODATA}, 0, A)]"

    # simple formula so just call command line
    call(
        [
            "gdal_calc.py",
            "-A",
            tid_tif,
            "--outfile",
            path,
            f"--calc={vs30}",
            f"--calc={stdv}",
            f"--NoDataValue={TERRAIN_NODATA}",
            "--type=Float32",
            "--co=COMPRESS=DEFLATE",
            "--quiet",
        ]
    )
    return path
