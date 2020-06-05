#!/usr/bin/env sh
#
# Make MBTiles for server upload as this gives better options
#
TIPPECANOE="/opt/tippecanoe/bin/tippecanoe"

$TIPPECANOE -l vspr -n "Measured Vs30 Sites" -Z0 -z14 -rg -o vspr.mbtiles vspr.geojson --force
$TIPPECANOE -l aak_map -n "AhdiAK Geology Category Map" -Z0 -z14 -rg -o aak_map.mbtiles aak_map.geojson --force
$TIPPECANOE -l ip_map -n "Iwahashi and Pike Terrain Category Map" -Z0 -z14 -rg -o ip_map.mbtiles iwahashipike.geojson --force
