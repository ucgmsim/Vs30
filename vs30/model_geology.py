import os

import numpy as np
from osgeo import gdal, ogr, osr

from vs30.model import interpolate, resample

gdal.UseExceptions()


QMAP = "qmap/qmap.shp"
COAST = "coast/nz-coastlines-and-islands-polygons-topo-1500k.shp"
SLOPE = "slope.tif"
ID_NODATA = 255
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
    names of levels
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
    return np.array([[161, 0.522],
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


def mid(points, args):
    """
    Faster method for model ID that uses rasterization.
    """
    gid_tif = mid_map(args)
    return interpolate(points, gid_tif)


def mid_polygon(points, modeldata):
    """
    Returns the category ID index (including 0 for water) for given locations.
    points: 2D numpy array of NZTM coords
    """
    shp = ogr.Open(os.path.join(modeldata, QMAP), gdal.GA_ReadOnly)
    lay = shp.GetLayer(0)
    col = lay.GetLayerDefn().GetFieldIndex("gid")

    values = np.empty(len(points), dtype=np.float32)
    # ocean is NaN while water polygons are 0
    values[...] = ID_NODATA
    pt = ogr.Geometry(ogr.wkbPoint)
    for i, p in enumerate(points):
        # why not decimal values??
        pt.AddPoint_2D(round(p[0]), round(p[1]))
        lay.SetSpatialFilter(pt)
        f = lay.GetNextFeature()
        if f is not None:
            values[i] = f.GetField(col)

    return values


def mid_map(args):
    """
    Optimised polygon search using geotiff rasterisation.
    """
    path = os.path.join(args.wd, "gid.tif")
    if os.path.isfile(path):
        return path
    # make sure output raster has a nicely defined projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2193)
    # calling Rasterize without saving result to variable will fail
    ds = gdal.Rasterize(
        path,
        os.path.join(args.mapdata, QMAP),
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
        outputSRS=srs,
        outputBounds=[args.xmin, args.ymin, args.xmax, args.ymax],
        xRes=args.xd,
        yRes=args.yd,
        noData=ID_NODATA,
        attribute="gid",
        outputType=gdal.GDT_Byte,
    )
    band = ds.GetRasterBand(1)
    band.SetDescription("Geology ID Index")
    band = None
    ds = None
    return path


def coast_distance_map(args):
    """
    Calculate coast distance needed for G06 and G13 mods.
    """
    path = os.path.join(args.wd, "coast.tif")
    # only need UInt16 (~65k max val) because coast only used 8->20k
    ds = gdal.Rasterize(
        path,
        os.path.join(args.mapdata, COAST),
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
        outputBounds=[args.xmin, args.ymin, args.xmax, args.ymax],
        xRes=args.xd,
        yRes=args.yd,
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
    return path


def slope_map(args):
    """
    Calculate slope at map points by resampling / resizing origin slope map.
    """
    dst = os.path.join(args.wd, "slope.tif")
    if os.path.isfile(dst):
        return dst
    src = os.path.join(args.mapdata, SLOPE)
    resample(src, dst, args.xmin, args.xmax, args.ymin, args.ymax, args.xd, args.yd)
    return dst


def model_map(args, model):
    path = os.path.join(args.wd, "geology.tif")
    # geology grid
    gid_tif = mid_map(args)
    gds = gdal.Open(gid_tif, gdal.GA_ReadOnly)
    g_band = gds.GetRasterBand(1)
    g_nodata = g_band.GetNoDataValue()
    # coastline distances
    if args.g6mod or args.g13mod:
        cdist_tif = coast_distance_map(args)
        cds = gdal.Open(cdist_tif, gdal.GA_ReadOnly)
        c_band = cds.GetRasterBand(1)
    if args.ghybrid:
        slope_tif = slope_map(args)
        sds = gdal.Open(slope_tif, gdal.GA_ReadOnly)
        s_band = sds.GetRasterBand(1)
        s_nodata = s_band.GetNoDataValue()
    # output
    driver = gdal.GetDriverByName("GTiff")
    ods = driver.Create(
        path,
        xsize=args.nx,
        ysize=args.ny,
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

    # model version for indexing (water id 0 and source NODATA -> NODATA)
    vs30 = np.append(MODEL_NODATA, model[:, 0]).astype(np.float32)
    stdv = np.append(MODEL_NODATA, model[:, 1]).astype(np.float32)
    if args.ghybrid:
        # sigma reduction factors
        stdv[HYBRID_SRF[0]] *= HYBRID_SRF[1]

    # processing chunk/block sizing
    block = band_vs30.GetBlockSize()
    nxb = (int)((args.nx + block[0] - 1) / block[0])
    nyb = (int)((args.ny + block[1] - 1) / block[1])

    for x in range(nxb):
        xoff = x * block[0]
        # last block may be smaller
        if x == nxb - 1:
            block[0] = args.nx - x * block[0]
        # reset y block size
        block_y = block[1]

        for y in range(nyb):
            yoff = y * block[1]
            # last block may be smaller
            if y == nyb - 1:
                block_y = args.ny - y * block[1]

            # determine results on block
            g_val = g_band.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
            )
            # treat ocean (out of polygon) as water (id 0)
            idx = np.where(g_val != g_nodata, g_val, 0)
            result_vs30 = vs30[idx]
            result_stdv = stdv[idx]
            if args.ghybrid:
                s_val = s_band.ReadAsArray(
                    xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
                )
                for spec in HYBRID_VS30:
                    if spec[0] == 4 and args.g6mod:
                        continue
                    w = np.where(g_val == spec[0])
                    # prevent -Inf warnings
                    s_val[np.where((s_val == 0) | (s_val == s_nodata))] = 1e-9
                    result_vs30[w] = 10 ** np.interp(
                        np.log10(s_val[w]), spec[1], spec[2]
                    )
            if args.g6mod or args.g13mod:
                c_val = c_band.ReadAsArray(
                    xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
                ).astype(np.float32)
                if args.g6mod:
                    result_vs30 = np.where(
                        g_val == 4,
                        np.maximum(
                            240,
                            np.minimum(
                                500, 240 + (500 - 240) * (c_val - 8000) / (20000 - 8000)
                            ),
                        ),
                        result_vs30,
                    )
                if args.g13mod:
                    result_vs30 = np.where(
                        g_val == 10,
                        np.maximum(
                            197,
                            np.minimum(
                                500, 197 + (500 - 197) * (c_val - 8000) / (20000 - 8000)
                            ),
                        ),
                        result_vs30,
                    )

            # write results
            band_vs30.WriteArray(result_vs30, xoff=xoff, yoff=yoff)
            band_stdv.WriteArray(result_stdv, xoff=xoff, yoff=yoff)
    # close
    band_vs30 = None
    band_stdv = None
    ods = None
    return path
