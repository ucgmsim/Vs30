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

## Grid based calculation


### Plotting


## Point based calculation


## Workflow
Vs30_r/run.R creates models from points or grid.
