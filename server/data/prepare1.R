#!/usr/bin/env Rscript
#
# Do not source with echo in Rstudio (click the source button)
# Prepares data for use with Mapbox
#
library(plotKML)
library(raster)

PREFIX = "/nesi/project/nesi00213/PlottingData/Vs30/"

AAK_MAP_OUT = "aak_map.geojson"
IP_OUT = "iwahashipike.tiff"

WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"


# AhdiAK Geology Category Map
load(paste0(PREFIX, "aak_map.Rdata"))
aak_map = spTransform(aak_map, WGS84)

sink(AAK_MAP_OUT, type="output")
cat('{"type":"FeatureCollection","features":[')

for (i in 1:length(aak_map@data$groupID_AhdiAK)) {
    gid = aak_map@data$groupID_AhdiAK[i]
    if (is.na(gid)) gid = 0
    if (i > 1) cat(',')
    cat(paste0('{"type":"Feature","properties":{"gid":', gid, '},"geometry":{"type":"Polygon","coordinates":['))
    polygons = aak_map@polygons[[i]]@Polygons
    for (p in 1:length(polygons)) {
        if (p == 1) {
            cat('[')
        } else {
            cat(',[')
        }
        coords = polygons[[p]]@coords
        for (r in 1:nrow(coords)) {
            if (r > 1) cat(',')
            cat(paste0('[',paste0(sprintf(coords[r,], fmt='%#.5f'), collapse=","), ']'))
        }
        cat(']')
    }
    cat(']}}')
}

cat(']}')
sink()


# IwahashiPike Terrain Map
#iwahashipike = as(raster(paste0(PREFIX, "IwahashiPike_NZ_100m_16.tif")), "SpatialGridDataFrame")
#iwahashipike = spTransform(iwahashipike, WGS84)
# use plotKML::reproject, regular grid not maintained with spTransform
#iwahashipike = plotKML::reproject(iwahashipike, WGS84)
#iwahashipike = raster(iwahashipike)
#writeRaster(iwahashipike, filename=IP_OUT, format="GTiff", overwrite=TRUE)
