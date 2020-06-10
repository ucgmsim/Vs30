#!/usr/bin/env Rscript

library(raster)

RASTER_DIR = "../../vs30out/"
OUT = "combined.geojson"
NZTM = "+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"


spdf = as(raster(paste0(RASTER_DIR, "vs30.tif")), "SpatialPointsDataFrame")
spdf_stdv = as(raster(paste0(RASTER_DIR, "stdev.tif")), "SpatialPointsDataFrame")
sink(OUT, type="output")
cat('{"type":"FeatureCollection","features":[')

for (i in 1:length(spdf@data[,])) {
    # SpatialPointsDataFrame already removes NA from grid
    vs30 = spdf@data[i, 1]
    stdv = spdf_stdv@data[i, 1]
    # spacing = 100m
    centre = spdf@coords[i, ]
    stopifnot(centre == spdf_stdv@coords[i, ])
    names(centre) = NULL
    x = c(centre[1] - 50, centre[1] + 50, centre[1] + 50, centre[1] - 50)
    y = c(centre[2] - 50, centre[2] - 50, centre[2] + 50, centre[2] + 50)
    points = data.frame(x, y)
    coordinates(points) = ~ x + y
    crs(points) = NZTM
    points = data.frame(spTransform(points, WGS84))
    
    if (i > 1) cat(',')
    cat(paste0('{"type":"Feature","properties":{"vs30":', vs30, ',"stdv":', stdv, '},"geometry":{"type":"Polygon","coordinates":[['))
    for (p in 1:length(points[, 1])) {
        if (p > 1) cat(',')
        cat(paste0('[',paste0(sprintf(points[p, ], fmt='%#.6f'), collapse=","), ']'))
    }
    cat(']]}}')
}

cat(']}')
sink()
