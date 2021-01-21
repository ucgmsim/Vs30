
import os

import numpy as np
from osgeo import gdal, ogr

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

    #geo_ref = lyr_in.GetSpatialRef()
    #point_ref=ogr.osr.SpatialReference()
    #point_ref.ImportFromEPSG(4326)
    #ctran=ogr.osr.CoordinateTransformation(point_ref,geo_ref)
    #[lon,lat,z]=ctran.TransformPoint(lon,lat)
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.SetPoint_2D(0, 1561153, 5171247)
    pt.SetPoint_2D(0, 1569329, 5164429)
    for p in points:
        pt.SetPoint_2D(0, int(p[0]), int(p[1]))
        lay.SetSpatialFilter(pt)
        for feat in lay:
            print(feat.GetField(col))

    points = np.array([[1561153, 5171247], [1568536, 5169878]])
    pts = ogr.Geometry(ogr.wkbMultiPoint)


    return values
    

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

points = points[:20000000]
pts = ogr.Geometry(ogr.wkbMultiPoint)
pt = ogr.Geometry(ogr.wkbPoint)
arr=np.empty(20000000)
arr[...] = np.nan
t1 = time()
for i, p in enumerate(points):
    pt.AddPoint_2D(int(p[0]), int(p[1]))
    #_=pts.AddGeometry(pt)
    lay.SetSpatialFilter(pt)
    f = lay.GetNextFeature()
    if f is not None:
        arr[i] = f.GetField(col)
t2=time()

# 13min / 20million points
# total 184 million = 117 minutes, 2 core hours


gids = gidx(points)
model = model_posterior_paper()
vals = gidx2val(model, gids)
save(vals[:, 0], x0, y0, nx, ny, xd, yd, "terrain.tif")
