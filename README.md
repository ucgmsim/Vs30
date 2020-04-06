# Vs30
Vs30 is used in the high frequency workflow.
Retrieve Vs30/standard deviation/residual at locations using Kevin Foster's research codes (modified).
Run for specific locations or over a grid.

## Setup
Install R on your system. Rstudio is an optional IDE.
There are a few R packages that the code requires. Packages that require system libraries such as rgdal require to be force re-installed with parent library ABI change.
```r
install.packages(c("raster", "rgdal", "gstat", "rgeos", "matrixcalc", "spatstat"))
```
Run the command in `R` as the `root` user for system-wide installation.

## Workflow
R scripts should be run with the working directory set as the repo root.
1. When measured site logic or data changes, re-run `./Kevin/vspr.R` which will update the vspr csv file.
1. Run grid or point based plotting (below).
1. Optionally create a plot for gridded outputs.

## Grid based calculation
`./run_grid.R` will run the calculation over a grid (edit parameters as required).

### Plotting
`./plot_map.py` will use GMT to make maps of the NZ-wide grid data. Run with `--help` for options.

## Point based calculation
`./run_points.R` will run the calculation over arbitrary points (edit parameters as required).
