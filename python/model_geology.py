
import os

import numpy as np
from osgeo import gdal, ogr, osr

PREFIX = "/run/media/vap30/Hathor/work/plotting_data/Vs30/"
QMAP = os.path.join(PREFIX, "qmap/qmap.shp")
COAST = os.path.join(PREFIX, "../Paths/lds-nz-coastlines-and-islands/EPSG_2193/nz-coastlines-and-islands-polygons-topo-1500k.shp")


def model_prior():
    """
    names of levels
       00_water (not used)
    0  01_peat
    1  04_fill
    2  05_fluvialEstuarine
    3  06_alluvium
    4  08_lacustrine
    5  09_beachBarDune
    6  10_fan
    7  11_loess
    8  12_outwash
    9  13_floodplain
    10 14_moraineTill
    11 15_undifSed
    12 16_terrace
    13 17_volcanic
    14 18_crystalline
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


def gidx(points):
    """
    Returns the category ID index (including 0 for water) for given locations.
    TODO: if there are many points, allow creating an intermediate raster.
    points: 2D numpy array of NZTM coords
    """
    shp = ogr.Open(QMAP, gdal.GA_ReadOnly)
    lay = shp.GetLayer(0)
    col = lay.GetLayerDefn().GetFieldIndex("gid")

    values = np.empty(len(points), dtype=np.float32)
    # ocean is NaN while water polygons are 0
    values[...] = np.nan
    pt = ogr.Geometry(ogr.wkbPoint)
    for i, p in enumerate(points):
        # why not decimal values??
        pt.AddPoint_2D(round(p[0]), round(p[1]))
        lay.SetSpatialFilter(pt)
        f = lay.GetNextFeature()
        if f is not None:
            values[i] = f.GetField(col)

    return values
    

def gidx_grid(filename, xmin, xmax, ymin, ymax, xd, yd):
    """
    Optimised polygon interpolation using geotiff rasterisation.
    """
    # make sure output raster has a nicely defined projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2193)
    # calling Rasterize without saving result to variable will fail
    ds = gdal.Rasterize(filename,
                        QMAP,
                        creationOptions=["COMPRESS=DEFLATE"],
                        outputSRS=srs,
                        outputBounds=[xmin, ymin, xmax, ymax],
                        xRes=xd,
                        yRes=yd,
                        noData=255,
                        attribute="gid",
                        outputType=gdal.GDT_Byte)
    ds = None


def coast_distance_grid(filename, xmin, xmax, ymin, ymax, xd, yd):
    # calling Rasterize without saving result to variable will fail
    ds = gdal.Rasterize(filename,
                        QMAP,
                        creationOptions=["COMPRESS=DEFLATE"],
                        outputBounds=[xmin, ymin, xmax, ymax],
                        xRes=xd,
                        yRes=yd,
                        noData=0,
                        burnValues=1,
                        outputType=gdal.GDT_Byte)
    ds = None
    # distance calculation
    # TODO: follow gdal_proximity.py coast.tif test.tif -co COMPRESS=DEFLATE -distunits GEO -ot Uint16 -values 0 -use_input_nodata YES
    ds = gdal.ComputeProximity(srcband, dstband, ["USE_INPUT_NODATA=YES"])
    ds = None
    srcband = None
    dstband = None


# grid setup
x0 = 1000000
xn = 2126400
y0 = 4700000
yn = 6338400
xd = 100
yd = 100
nx = round((xn - x0) / xd + 1)
ny = round((yn - y0) / yd + 1)
# run
gids = gidx_grid("geology.tif", x0, xn, y0, yn, xd, yd)
test = coast_distance_grid("coast.tif", x0, xn, y0, yn, xd, yd)
model = model_posterior_paper()
vals = gidx2val(model, gids)
save(vals[:, 0], x0, y0, nx, ny, xd, yd, "terrain.tif")
