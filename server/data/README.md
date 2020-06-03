# Data preparation for upload to Mapbox

## Step 1: Convert datasets to be understood by tippecanoe
```
./prepare1.R
```

## Step 2: Create mbtiles with tippecanoe
This allows more flexibility by being able to specify zoom levels where data is available amongst other things.
```
./prepare2.sh
```

## Step 3: Upload data to Mapbox
Upload the mbtiles to the tilesets section in your account.

## Step 4: Create a map on the server
`Satellite Streets` is a good basemap to work with because it has a single baselayer with streets/labels above.
Add layers above the baselayer but below all the roads/labels/etc.

Because there are 16 `aak_map` categories (0-15), that makes for a spacing of 22.5 on the colour circle. Currently you can set the colour fill formula as `"hsl(" & get("gid") * 22.5 & ", 100%, 75%)"`. Unlike `rgb`, the `hsl` specification at time of writing needs to be a string.

## Step 5: Udate references
The layers in the map you create have names and these should be updated in the Javascript. Also the map itself has an ID which should be updated.
