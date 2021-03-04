# Vs30
Vs30 is used in the high frequency workflow.
Retrieve Vs30/standard deviation/residual at locations using Kevin Foster's research codes (modified).
Run for specific locations or over a grid.

## Setup
Make sure you have Python 3 on your system.
There are a few Python packages that the code requires. Packages that require system libraries such as `gdal` require to be re-compiled with parent library ABI change. This is especially important when using `pip` because it doesn't manage binary dependencies. `gdal` should be installed using the system package manager.

You can either install the packages via the system package manager or using pip. The list of packages is:
 * GDAL (requires gdal system libraries, should be installed as system package, import is osgeo)
 * pandas
 * numpy
 * pyproj (requires proj system library)
 * scikit-learn (for measured site clustering)
 * scipy (for downsampling the McGann dataset)

Install the package (optional):
```shell
pip install --user .
```
You may have to add the installed script location to `PATH` if you cannot run `vs30calc.py` from other locations:
```shell
export PATH=$PATH:$HOME/.local/bin
```

If you aren't using the installed package, you will need to extract the compressed shapefiles (`vs30/data/shapefiles.tar.xz`) such that the folders `vs30/data/coast` and `vs30/data/qmap` exist.

## Workflow
Everything is run from the `vs30calc.py` script which can be run directly or installed.
The 2 main modes of operation are grid based (which creates TIF files) and point based (where wanted locations are passed in as rows in a file, output is a CSV file). Everything is NZTM2000 (EPSG:2193, easting/northing) internally, WGS84 (EPSG:4326, longitude/latitude) is only used as input to the point based calculation.

Both modes produce `.qgz` files which can be opened in `QGIS` to view outputs.

## Grid based calculation
This mode is good for viewing the entire model, making sure everything appears good. This is the default mode of operation if no locations file is specified.
The grid has parameters which control the extent and spacing.
`vs30calc.py --help`
A sample command for creating a smaller grid (~5 mins depending on machine):
```shell
vs30calc.py --xmin 1470050 --xmax 1580050 --ymin 5150050 --ymax 5250050
```

## Point based calculation
This mode is much faster because it only calculates the model values at given locations, not at millions of points around the country.
Specify a file containing locations with `--ll-path`. The default is no header to skip (`--skip-rows 0`), longitude in the first column (`--lon-col-ix 0`), latitude in the second (`--lat-col-ix 1`) and space as the column separator (`--col-sep " "`). There are options for this as well.
Note that grids are used for some intermediate parts of the calculation in this mode so the parameters are still applicable.
