#!/usr/bin/env Rscript
#
# Do not source with echo in Rstudio (click the source button)
# Prepares data for use with Mapbox
#
library(sp)
library(raster)

PREFIX = "/nesi/project/nesi00213/PlottingData/Vs30/"

AAK_MAP_OUT = "aak_map.geojson"
IP_OUT = "iwahashipike.geojson"
VSPR_OUT = "vspr.geojson"

NZTM = "+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
WEBMERC = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs"


# measured sites
vspr = read.csv("../../data/vspr.csv")
sink(VSPR_OUT, type="output")
cat('{"type":"FeatureCollection","features":[')

for (i in seq_len(nrow(vspr))) {
    # values may be NA if unavailable
    quality_flag = vspr[i, "QualityFlag"]
    station_id = vspr[i, "StationID"]
    Vs30_AhdiAK_noQ3_hyb09c = vspr[i, "Vs30_AhdiAK_noQ3_hyb09c"]
    stDv_AhdiAK_noQ3_hyb09c = vspr[i, "stDv_AhdiAK_noQ3_hyb09c"]
    Vs30_YongCA_noQ3 = vspr[i, "Vs30_YongCA_noQ3"]
    stDv_YongCA_noQ3 = vspr[i, "stDv_YongCA_noQ3"]
    if (is.na(quality_flag)) quality_flag = ""
    if (is.na(station_id)) station_id = ""
    if (is.na(Vs30_AhdiAK_noQ3_hyb09c)) Vs30_AhdiAK_noQ3_hyb09c = "[]"
    if (is.na(stDv_AhdiAK_noQ3_hyb09c)) stDv_AhdiAK_noQ3_hyb09c = "[]"
    if (is.na(Vs30_YongCA_noQ3)) Vs30_YongCA_noQ3 = "[]"
    if (is.na(stDv_YongCA_noQ3)) stDv_YongCA_noQ3 = "[]"

    if (i > 1) cat(',')
    cat(paste0('{"type":"Feature","properties":{"Easting":', vspr[i, "x"], 
            ',"Northing":', vspr[i, "y"],
            ',"Vs30":', vspr[i, "Vs30"],
            ',"lnMeasUncer":', vspr[i, "lnMeasUncer"],
            ',"QualityFlag":"', quality_flag,
            '","StationID":"', station_id,
            '","Vs30_AhdiAK_noQ3_hyb09c":', Vs30_AhdiAK_noQ3_hyb09c,
            ',"stDv_AhdiAK_noQ3_hyb09c":', stDv_AhdiAK_noQ3_hyb09c,
            ',"Vs30_YongCA_noQ3":', Vs30_YongCA_noQ3,
            ',"stDv_YongCA_noQ3":', stDv_YongCA_noQ3,
            '},"geometry":{"type":"Point","coordinates":[', 
            vspr[i, "longitude"], ',', vspr[i, "latitude"], ']}}'))
}

cat(']}')
sink()


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
iwahashipike = as(raster(paste0(PREFIX, "IwahashiPike_NZ_100m_16.tif")), "SpatialPointsDataFrame")
sink(IP_OUT, type="output")
cat('{"type":"FeatureCollection","features":[')

# plotKML::reproject has issues and also polygons are more accurate
for (i in 1:length(iwahashipike@data[,])) {
    # SpatialPointsDataFrame already removes NA from grid
    gid = iwahashipike@data[i, 1]
    # spacing = 100m
    centre = iwahashipike@coords[i, ]
    names(centre) = NULL
    x = c(centre[1] - 50, centre[1] + 50, centre[1] + 50, centre[1] - 50)
    y = c(centre[2] - 50, centre[2] - 50, centre[2] + 50, centre[2] + 50)
    points = data.frame(x, y)
    coordinates(points) = ~ x + y
    crs(points) = NZTM
    points = data.frame(spTransform(points, WGS84))

    if (i > 1) cat(',')
    cat(paste0('{"type":"Feature","properties":{"gid":', gid, '},"geometry":{"type":"Polygon","coordinates":[['))
    for (p in 1:length(points[, 1])) {
        if (p > 1) cat(',')
        cat(paste0('[',paste0(sprintf(points[p, ], fmt='%#.5f'), collapse=","), ']'))
    }
    cat(']]}}')
}

cat(']}')
sink()
