# Vs30
Vs30 is used in the high frequency workflow.
Retrieve Vs30/standard deviation/residual at locations using Kevin Foster's research codes (modified).
Run for specific locations or over a grid.

## Setup
Make sure you have Python 3 on your system.
There are a few Python packages that the code requires. Packages that require system libraries such as `gdal` require to be re-compiled with parent library ABI change. This is especially important when using `pip` because it doesn't manage binary dependencies.

You can either install the packages via the system package manager or using pip. The list of packages is:
 * pandas
 * numpy
 * pyproj (requires proj system library)
 * pygdal (requires gdal system library, may be included as part of gdal, import is osgeo)
 * sklearn (for measured site clustering)
 * scipy (for downsampling the McGann dataset)

Make sure you have the large data files available in the `mapdata` argument, to prevent specifying on every run, change the default (`PREFIX` in `vs30/params.py`).

## Workflow
Everything is run from the vs30calc.py script which can be run directly or installed.
If you have installed the library outside an environment under your user account, you may have to add the location to the `PATH`:
```shell
export PATH=$PATH:$HOME/.local/bin
```
The 2 main modes of operation are grid based (which creates TIF files) and point based (where wanted locations are passed in).

## Grid based calculation
This mode is good for viewing the entire model, making sure everything appears good. TIF files can be viewed with QGIS. This is the default mode of operation if no locations file is specified.
The grid has parameters which control the extent and spacing.
`vs30calc.py --help`

## Point based calculation
This mode is much faster because it only calculates the model values at given locations, not at millions of points around the country.
Specify a file containing locations with `--ll-path`. The default is no header to skip, longitude in the first column and latitude in the second. There are options for this as well.
