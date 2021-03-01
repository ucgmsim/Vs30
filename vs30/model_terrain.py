"""
Terrain model functions based on IwahashiPike terrain categories and YongCA values.
"""
import os
from subprocess import call

import numpy as np
from osgeo import gdal

from vs30.model import ID_NODATA, interpolate_raster, resample_raster

gdal.UseExceptions()

# input location
MODEL_RASTER = os.path.join(os.path.dirname(__file__), "data", "IwahashiPike.tif")
# for output model
MODEL_NODATA = -32767


def model_prior():
    """
    ID in datasource, name of level
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


def model_id(points):
    """
    Returns the category ID index for given locations.
    points: 2D numpy array of NZTM coords
    """
    return interpolate_raster(points, MODEL_RASTER)


def model_id_map(paths, grid):
    """
    Calculate id at map points by resampling / resizing origin id map.
    """
    dst = os.path.join(paths.out, "tid.tif")
    if os.path.isfile(dst):
        return dst
    resample_raster(
        MODEL_RASTER, dst, grid.xmin, grid.xmax, grid.ymin, grid.ymax, grid.dx, grid.dy
    )
    r = gdal.Open(dst)
    b = r.GetRasterBand(1)
    b.SetDescription("Terrain ID Index")
    b = None
    r = None
    return dst


def model_val(ids, model, opts, paths=None, points=None, grid=None):
    """
    Return model values for IDs (vs30, stdv).
    """
    # allow just giving locations if ids not wanted
    if ids is None:
        ids = model_id(points)

    idx = ids != ID_NODATA
    result = np.full((len(ids), 2), np.nan, dtype=np.float32)
    result[idx] = model[ids[idx] - 1]
    return result


def model_val_map(paths, grid, model, opts):
    """
    Make a tif map of model values.
    """
    path = os.path.join(paths.out, "terrain.tif")
    # terrain IDs for given map spec
    tid_tif = model_id_map(paths, grid)
    raster = gdal.Open(tid_tif, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    band = None
    raster = None

    # model version for indexing (index 0 for NODATA)
    vs30 = np.append(MODEL_NODATA, model[:, 0]).astype(np.float32)
    stdv = np.append(MODEL_NODATA, model[:, 1]).astype(np.float32)
    # string array
    vs30 = ",".join([f"{x:.8f}" for x in vs30])
    stdv = ",".join([f"{x:.8f}" for x in stdv])
    # string expression to pass into calc
    vs30 = (
        f"numpy.array([{vs30}], dtype=numpy.float32)[numpy.where(A == {nodata}, 0, A)]"
    )
    stdv = (
        f"numpy.array([{stdv}], dtype=numpy.float32)[numpy.where(A == {nodata}, 0, A)]"
    )

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
            f"--NoDataValue={MODEL_NODATA}",
            "--type=Float32",
            "--co=COMPRESS=DEFLATE",
            "--co=BIGTIFF=YES",
            "--quiet",
        ]
    )
    # name bands
    raster = gdal.Open(path)
    bvs30 = raster.GetRasterBand(1)
    bstdv = raster.GetRasterBand(2)
    bvs30.SetDescription("Vs30")
    bstdv.SetDescription("Standard Deviation")
    bvs30 = None
    bstdv = None
    raster = None

    return path
