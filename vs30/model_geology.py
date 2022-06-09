"""
Geology model functions based on GNS QMAP geology categories and AhdiAk values.
"""
from copy import deepcopy
import os

import numpy as np
from osgeo import gdal, ogr, osr

from vs30.model import ID_NODATA, interpolate_raster, resample_raster

gdal.UseExceptions()

data = os.path.join(os.path.dirname(__file__), "data")
QMAP = os.path.join(data, "qmap", "qmap.shp")
COAST = os.path.join(data, "coast", "nz-coastlines-and-islands-polygons-topo-1500k.shp")
SLOPE = os.path.join(data, "slope.tif")
SLOPE_NODATA = -9999
MODEL_NODATA = -32767
# hybrid model vs30 based on interpolation of slope
# group ID, log10(slope) array, vs30 array
HYBRID_VS30 = [
    [2, [-1.85, -1.22], np.log10(np.array([242, 418]))],
    [3, [-2.70, -1.35], np.log10(np.array([171, 228]))],
    [4, [-3.44, -0.88], np.log10(np.array([252, 275]))],
    [6, [-3.56, -0.93], np.log10(np.array([183, 239]))],
]
# hybrid model sigma reduction factors
HYBRID_SRF = np.array([2, 3, 4, 6]), np.array([0.4888, 0.7103, 0.9988, 0.9348])


def model_prior():
    """
    ID NAME (id in datasource)
    0  00_water (not used)
    1  01_peat
    2  04_fill
    3  05_fluvialEstuarine
    4  06_alluvium
    5  08_lacustrine
    6  09_beachBarDune
    7  10_fan
    8  11_loess
    9  12_outwash
    10 13_floodplain
    11 14_moraineTill
    12 15_undifSed
    13 16_terrace
    14 17_volcanic
    15 18_crystalline
    """
    # fmt: off
    return np.array([[161, 0.522], # peat
                     [198, 0.314],
                     [239, 0.867],
                     [323, 0.365],
                     [326, 0.135],
                     [339, 0.647],
                     [360, 0.338],
                     [376, 0.380],
                     [399, 0.305],
                     [448, 0.432],
                     [453, 0.512],
                     [455, 0.545],
                     [458, 0.761],
                     [635, 0.995],
                     [750, 0.641]], dtype=np.float32)
    # fmt: on


def model_posterior_paper():
    """
    Posterior model from the paper.
    """
    # fmt: off
    return np.array([[162.892120019443, 0.301033220758108],
                     [272.512711921894, 0.280305955290694],
                     [199.502400319993, 0.438753673163297],
                     [271.050623687222, 0.243486579272276],
                     [326, 0.5],
                     [204.373998521848, 0.232196981798137],
                     [246.6138371423, 0.344601218802256],
                     [472.749388885179, 0.354562104171167],
                     [399, 0.5],
                     [197.472849225266, 0.202629769603115],
                     [453, 0.512],
                     [455, 0.545],
                     [335.279503497185, 0.602886888230288],
                     [635, 0.995],
                     [690.974348966211, 0.446036993981441]], dtype=np.float32)
    # fmt: on


def model_id(points):
    """
    Returns the category ID index (including 0 for water) for given locations.
    points: 2D numpy array of NZTM coords
    """
    shp = ogr.Open(QMAP, gdal.GA_ReadOnly)
    lay = shp.GetLayer(0)
    col = lay.GetLayerDefn().GetFieldIndex("gid")

    # ocean is NaN while water polygons are 0
    values = np.full(len(points), ID_NODATA, dtype=np.uint8)
    pt = ogr.Geometry(ogr.wkbPoint)
    # pt.AddPoint_2D will not work with float32, float64 is fine
    for i, p in enumerate(points.astype(np.float64)):
        pt.AddPoint_2D(p[0], p[1])
        lay.SetSpatialFilter(pt)
        f = lay.GetNextFeature()
        if f is not None:
            values[i] = f.GetField(col)

    return values


def model_id_fast(points, paths, grid):
    """
    Faster version of model_id that uses rasterisation instead of polygons.
    points: 2D numpy array of NZTM coords
    """
    gid_tif = model_id_map(paths, grid)
    return interpolate_raster(points, gid_tif)


def model_id_map(paths, grid):
    """
    Optimised polygon search using geotiff rasterisation.
    """
    path = os.path.join(paths.out, "gid.tif")
    if os.path.isfile(path):
        return path
    # make sure output raster has a nicely defined projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2193)
    # calling Rasterize without saving result to variable will fail
    ds = gdal.Rasterize(
        path,
        QMAP,
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
        outputSRS=srs,
        outputBounds=[grid.xmin, grid.ymin, grid.xmax, grid.ymax],
        xRes=grid.dx,
        yRes=grid.dy,
        noData=ID_NODATA,
        attribute="gid",
        outputType=gdal.GDT_Byte,
    )
    band = ds.GetRasterBand(1)
    band.SetDescription("Geology ID Index")
    band = None
    ds = None
    return path


def _full_land_grid(grid):
    """
    Extends grid for full land coverage.
    """
    landgrid = deepcopy(grid)
    gridmod = False
    if grid.xmin > 1060050:
        landgrid.xmin -= np.ceil((grid.xmin - 1060050) / grid.dx) * grid.dx
        gridmod = True
    if grid.xmax < 2120050:
        landgrid.xmax += np.ceil((2120050 - grid.xmax) / grid.dx) * grid.dx
        gridmod = True
    if grid.ymin > 4730050:
        landgrid.ymin -= np.ceil((grid.ymin - 4730050) / grid.dy) * grid.dy
        gridmod = True
    if grid.ymax < 6250050:
        landgrid.ymax += np.ceil((6250050 - grid.ymax) / grid.dy) * grid.dy
        gridmod = True

    return landgrid, gridmod


def coast_distance_map(paths, grid):
    """
    Calculate coast distance needed for G06 and G13 mods.
    """
    path = os.path.join(paths.out, "coast.tif")
    if os.path.isfile(path):
        return path

    # algorithm requires full land coverage
    landgrid, gridmod = _full_land_grid(grid)
    # only need UInt16 (~65k max val) because coast only used 8->20k
    ds = gdal.Rasterize(
        path,
        COAST,
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
        outputBounds=[landgrid.xmin, landgrid.ymin, landgrid.xmax, landgrid.ymax],
        xRes=landgrid.dx,
        yRes=landgrid.dy,
        noData=0,
        burnValues=1,
        outputType=gdal.GetDataTypeByName("UInt16"),
    )
    # distance calculation from outside polygons (0 value)
    band = ds.GetRasterBand(1)
    band.SetDescription("Distance to Coast (m)")
    # ComputeProximity doesn't respect any NODATA options (writing into self though)
    ds = gdal.ComputeProximity(band, band, ["VALUES=0", "DISTUNITS=GEO"])
    band = None
    ds = None

    if gridmod:
        # had to extend for land coverage, cut down to wanted size
        resample_raster(
            path, path, grid.xmin, grid.xmax, grid.ymin, grid.ymax, grid.dx, grid.dy
        )

    return path


def slope_map(paths, grid):
    """
    Calculate slope at map points by resampling / resizing origin slope map.
    """
    dst = os.path.join(paths.out, "slope.tif")
    if os.path.isfile(dst):
        return dst
    resample_raster(
        SLOPE, dst, grid.xmin, grid.xmax, grid.ymin, grid.ymax, grid.dx, grid.dy
    )
    return dst


def _hyb_calc(
    geol, model, gid, slope=None, cdist=None, nan=np.nan, s_nodata=SLOPE_NODATA
):
    """
    Return model values given inputs.
    """
    # sigma reduction factors
    if geol.hybrid:
        model = np.copy(model)
        # still updating g06 even if using coast function instead of hybrid
        model[HYBRID_SRF[0] - 1, 1] *= HYBRID_SRF[1]
    # output
    vs30 = np.full(gid.shape, nan, dtype=np.float32)
    stdv = np.copy(vs30)
    # water points are NaN
    w_mask = (gid != ID_NODATA) & (gid != 0)
    vs30[w_mask], stdv[w_mask] = model[gid[w_mask] - 1].T
    if geol.hybrid:
        # prevent -Inf warnings
        slope[np.where((slope == 0) | (slope == s_nodata))] = 1e-9
        for spec in HYBRID_VS30:
            if spec[0] == 4 and geol.mod6:
                continue
            w_mask = np.where(gid == spec[0])
            vs30[w_mask] = 10 ** np.interp(np.log10(slope[w_mask]), spec[1], spec[2])
    if geol.mod6:
        w_mask = np.where(gid == 4)
        # explicitly set cdist as float32 which can propagate through
        # keeping it as uint16 causes overflows with integer multiplication
        vs30[w_mask] = np.maximum(
            240,
            np.minimum(
                500,
                240
                + (500 - 240)
                * (cdist[w_mask].astype(np.float32) - 8000)
                / (20000 - 8000),
            ),
        )
    if geol.mod13:
        w_mask = np.where(gid == 10)
        vs30[w_mask] = np.maximum(
            197,
            np.minimum(
                500,
                197
                + (500 - 197)
                * (cdist[w_mask].astype(np.float32) - 8000)
                / (20000 - 8000),
            ),
        )

    return vs30, stdv


def model_val(ids, model, opts, paths=None, points=None, grid=None):
    """
    Return model values for IDs (vs30, stdv).
    """
    # collect inputs
    ids = model_id(points) if ids is None else ids
    cdist, slope = None, None
    # coastline distances and slope rough enough to keep as rasters (for now)
    if opts.mod6 or opts.mod13:
        cdist = interpolate_raster(points, coast_distance_map(paths, grid))
    if opts.hybrid:
        slope = interpolate_raster(points, slope_map(paths, grid))
    # run
    return np.column_stack(_hyb_calc(opts, model, ids, slope=slope, cdist=cdist))


def model_val_map(paths, grid, model, opts):
    """
    Make a tif map of model values.
    """
    path = os.path.join(paths.out, "geology.tif")
    # geology grid
    gid_tif = model_id_map(paths, grid)
    gds = gdal.Open(gid_tif, gdal.GA_ReadOnly)
    g_band = gds.GetRasterBand(1)
    # coastline distances
    c_val = None
    if opts.mod6 or opts.mod13:
        cdist_tif = coast_distance_map(paths, grid)
        cds = gdal.Open(cdist_tif, gdal.GA_ReadOnly)
        c_band = cds.GetRasterBand(1)
    s_val = None
    if opts.hybrid:
        slope_tif = slope_map(paths, grid)
        sds = gdal.Open(slope_tif, gdal.GA_ReadOnly)
        s_band = sds.GetRasterBand(1)
    # output
    driver = gdal.GetDriverByName("GTiff")
    ods = driver.Create(
        path,
        xsize=grid.nx,
        ysize=grid.ny,
        bands=2,
        eType=gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
    )
    ods.SetGeoTransform(gds.GetGeoTransform())
    ods.SetProjection(gds.GetProjection())
    band_vs30 = ods.GetRasterBand(1)
    band_stdv = ods.GetRasterBand(2)
    band_vs30.SetDescription("Vs30")
    band_stdv.SetDescription("Standard Deviation")
    band_vs30.SetNoDataValue(MODEL_NODATA)
    band_stdv.SetNoDataValue(MODEL_NODATA)

    # processing chunk/block sizing
    block = band_vs30.GetBlockSize()
    nxb = (int)((grid.nx + block[0] - 1) / block[0])
    nyb = (int)((grid.ny + block[1] - 1) / block[1])

    for x in range(nxb):
        xoff = x * block[0]
        # last block may be smaller
        if x == nxb - 1:
            block[0] = grid.nx - x * block[0]
        # reset y block size
        block_y = block[1]

        for y in range(nyb):
            yoff = y * block[1]
            # last block may be smaller
            if y == nyb - 1:
                block_y = grid.ny - y * block[1]

            # determine results on block
            g_val = g_band.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
            )
            if opts.hybrid:
                s_val = s_band.ReadAsArray(
                    xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
                )
            if opts.mod6 or opts.mod13:
                c_val = c_band.ReadAsArray(
                    xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
                ).astype(np.float32)
            result_vs30, result_stdv = _hyb_calc(
                opts, model, g_val, slope=s_val, cdist=c_val, nan=MODEL_NODATA
            )
            # write results
            band_vs30.WriteArray(result_vs30, xoff=xoff, yoff=yoff)
            band_stdv.WriteArray(result_stdv, xoff=xoff, yoff=yoff)
    # close
    band_vs30 = None
    band_stdv = None
    ods = None
    return path
