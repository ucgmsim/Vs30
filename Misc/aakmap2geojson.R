#!/usr/bin/env Rscript
#
# Do not source with echo in Rstudio (click the source button)
#

PREFIX = "/nesi/project/nesi00213/PlottingData/Vs30/"
OUT = "aak_map.geojson"

WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

load(paste0(PREFIX, "aak_map.Rdata"))
aak_map = spTransform(aak_map, WGS84)

sink(OUT, type="output")
cat('{"type":"FeatureCollection","features":[')

for (i in 1:length(aak_map@data$groupID_AhdiAK)) {
    gid = aak_map@data$groupID_AhdiAK[i]
    if (i > 1) cat(',')
    cat(paste0('{"properties":{"g":"', gid, '"},"geometry":{"type":"Polygon","coordinates":['))
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
