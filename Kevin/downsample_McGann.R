library('raster')
library('spatstat')


downsample_spdf = function(inputSPDF, res=1000) {
  # resample SpatialPointsDataFrame on res [m] grid

  max_dist = sqrt(2 * res^2) / 2
  # grid locations using raster::raster()
  g = as(raster::raster(crs=proj4string(inputSPDF), ext=extent(bbox(inputSPDF)), resolution=res), "SpatialPoints")

  # nncross takes 2 point patterns
  grd = spatstat::ppp(x=coordinates(g)[,1], y=coordinates(g)[,2],
                      window=owin(xrange=bbox(g)[1,], yrange=bbox(g)[2,]))
  inppp = spatstat::ppp(x=coordinates(inputSPDF)[,1], y=coordinates(inputSPDF)[,2],
                        window=owin(xrange=bbox(inputSPDF)[1,], yrange=bbox(inputSPDF)[2,]))
  distz = spatstat::nncross(grd, inppp)
  # no further than half the grid hypotenuse
  distz = distz[distz$dist < max_dist,]

  # Remove all other points from input dataframe
  return(inputSPDF[distz$which,])
}
