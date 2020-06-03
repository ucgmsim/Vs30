#!/usr/bin/env sh
#
# Make MBTiles for server upload as this gives better options
#
TIPPECANOE="/opt/tippecanoe/bin/tippecanoe"

$TIPPECANOE -l aak_map -n "AhdiAK Geology Category Map" -Z3 -z14 -o aak_map.mbtiles aak_map.geojson
