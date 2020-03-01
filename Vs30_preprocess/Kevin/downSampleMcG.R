
# Note: committed before shutdown 20171206

# Make tooFar accessible to other scripts that source this one
# (i.e. downSampleMcG_plot.R)
# tooFar  <-  200 # meters
# tooFar  <-  500 # meters
tooFar  <-  707.2 # meters



downSampleMcG <- function(inputSPDF) {
  # ===========================================================================
  # resample McGann points on 1km grid
  
  
  # Resampling requires using "nncross" from package "spatstat"
  # which in turn requires converting points data to class "ppp"...
  library('raster')
  library('spatstat')
  

  
  gridRes <- 1000 # meters
  
  
  # run once to examine effects of various resampling densities for McGann data
  # results are saved in Vs30_tables and Rdata/ folders under dated files
  # with prefix 20170410
  #gridRes <- 3000 # meters
  #gridRes <- 4000 # meters
  #gridRes <- 5000 # meters 
  
  

  
  

  
  
  # Make the grid. Use raster() and then convert.
  # Not the most direct but it works.
  
  ggg <-raster(crs = proj4string(inputSPDF),
               ext = extent(bbox(inputSPDF)),
               resolution = gridRes)
  gg <- as(ggg, 'SpatialGrid')
  g <- as(gg, 'SpatialPoints')
  
  
  # Note that with 1000 meter spacing, a grid overlay on McGann points
  # ends up as 21 x 43 km, with 22x44 points = 968 points:
  # > points2grid(gg)
  # s1      s2
  # cellcentre.offset 2468048 5725866
  # cellsize             1000    1000
  # cells.dim              22      44
  
  # > dim(coordinates(g))
  # [1] 968   2
  
  
  # converting to "ppp"...
  grd <- ppp(x = coordinates(g)[,1],
             y = coordinates(g)[,2],
             window = owin(xrange=bbox(g)[1,], yrange = bbox(g)[2,]))
  
  inpSPDFppp <- ppp(x = coordinates(inputSPDF)[,1],
                y = coordinates(inputSPDF)[,2],
                window = owin(xrange=bbox(inputSPDF)[1,], yrange=bbox(inputSPDF)[2,]))
  
  # Finally, running nncross
  distz <- nncross(grd, inpSPDFppp) # each row corresponds to one of the grid points.
  
  # Just to make sure, check that the coordinates of grd are in same order as those in g...
  # dd <- 47 # random coordinate
  # grd$x[dd]
  # grd$y[dd]
  # coordinates(g)[dd,]
  # ^ ok, they match
  
  # I want to take only the points in McGann original data that are less than "tooFar"
  # distance away from one of the grid points.
  
  
  # Throw out all data too far from a McGann point.
  distz <- distz[distz$dist < tooFar,]
  
  
  # Remove all other points from input dataframe
  outp <- inputSPDF[distz$which,] 
  
  return(outp)
  
  # to plot the resampling of McGann's data, use downSampleMcG_plot.R
}



