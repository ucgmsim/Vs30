
import os

import numpy as np
from osgeo import gdal, ogr, osr

PREFIX = "/run/media/vap30/Hathor/work/plotting_data/Vs30/"
QMAP = os.path.join(PREFIX, "qmap/qmap.shp")


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
                        targetAlignedPixels=False,
                        attribute="gid",
                        outputType=gdal.GDT_Byte)
    ds = None



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
gids = gidx_grid("geology.tiff", x0, xn, y0, yn, xd, yd)
model = model_posterior_paper()
vals = gidx2val(model, gids)
save(vals[:, 0], x0, y0, nx, ny, xd, yd, "terrain.tif")
