
import os

import numpy as np
from osgeo import gdal, ogr, osr

PREFIX = "/run/media/vap30/Hathor/work/plotting_data/Vs30/"
QMAP = os.path.join(PREFIX, "qmap/qmap.shp")


def gidx(points):
    """
    Returns the category ID index for given locations. 255 for NaN.
    points: 2D numpy array of NZTM coords
    """
    drv = ogr.GetDriverByName("ESRI Shapefile")
    shp = drv.Open(QMAP, gdal.GA_ReadOnly)
    lay = shp.GetLayer(0)
    col = lay.GetLayerDefn().GetFieldIndex("gid")

    values = np.empty(len(points), dtype=np.float32)
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
    

def gidx_grid(xmin, xmax, ymin, ymax, xd, yd):
    """
    Optimised polygon interpolation using geotiff rasterisation.
    """
    xmin = 1000000
    xmax = 2126400
    ymin = 4700000
    ymax = 6338400
    xd = 100
    yd = 100
    width = round((xmax - xmin) / xd)
    height = round((ymax - ymin) / yd)
    shp = ogr.Open(QMAP, gdal.GA_ReadOnly)
    lay = shp.GetLayer(0)
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2193)
    ds = gdal.Rasterize("new.tif",
                        QMAP,
                        outputSRS=srs,
                        outputBounds=[xmin, ymin, xmax, ymax],
                        xRes=xd,
                        yRes=yd,
                        #noData=255,
                        targetAlignedPixels=True,
                        attribute="gid",
                        outputType=gdal.GDT_Float32)
    

# grid setup
x0 = 1000050
xn = 2126350
y0 = 4700050
yn = 6338350
xd = 100
yd = 100
nx = round((xn - x0) / xd + 1)
ny = round((yn - y0) / yd + 1)



exit()
# run
points = np.vstack(np.mgrid[x0:xn + 1:xd,y0:yn + 1:yd].T)
gids = gidx(points)
model = model_posterior_paper()
vals = gidx2val(model, gids)
save(vals[:, 0], x0, y0, nx, ny, xd, yd, "terrain.tif")
