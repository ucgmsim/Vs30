# R
R version of the Vs30 workflow.

## Setup
Install R on your system. Rstudio is an optional IDE.
There are a few R packages that the code requires. Packages that require system libraries such as rgdal require to be force re-installed with parent library ABI change.
```r
install.packages(c("raster", "rgdal", "gstat", "rgeos", "matrixcalc", "spatstat", "ncdf4", "proxy"))
```
Run the command in `R` as the `root` user for system-wide installation.

Make sure you have the large data files available in the `PREFIX` definition of `config.R`.

## Workflow
R scripts should be run with the working directory set as the repo root.
1. Run grid, point based or interactive web calculation (below).
1. Optionally create a plot for gridded outputs.

## Grid based calculation
`./run_grid.R` will run the calculation over a grid (edit parameters as required).

### Plotting
`./plot_map.py` will use GMT to make maps of the NZ-wide grid data. Run with `--help` for options.

## Point based calculation
`./run_points.R` will run the calculation over arbitrary points (edit parameters as required).

## Interactive web app
See [server](../server) for instructions.