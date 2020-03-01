# geodetic.R

# Just geodetic conversions. For applying to objects of type sp*
# (spatialPointsDataFrame, spatialPolygons, etc.)

# Note to self - NZ coordinate systems described here
# http://gis.stackexchange.com/questions/20389/converting-nzmg-or-nztm-to-latitude-longitude-for-use-with-r-map-library

library(sp)

convert2NZMG <- function(inp) {
  NZMGdata  <- spTransform(inp, CRS=crsNZMG())
  return(NZMGdata)
}

convert2NZGD49 <- convert2NZMG

convert2NZGD00 <- function(inp) {
  NZGD00data <- spTransform(inp, crsNZGD00())
  return(NZGD00data)
}

convert2WGS84 <- function(inp) {
  WGSdata <- spTransform(inp, crsWGS84())
  return(WGSdata)
} 


crsNZMG <- function(){
  return(CRS(paste0(
    "+proj=nzmg +lat_0=-41 +lon_0=173 +x_0=2510000 +y_0=6023150 ",
#   "+datum=nzgd49 +units=m +no_defs +ellps=intl ",
                  "+units=m +no_defs +ellps=intl ",  # months after originally writing this, scripts are behaving badly.
                                                     # over() function in vspr.R is the culprit. Removing "nzgd49" tag is
                                                     # an attempt to solve this problem.
    "+towgs84=59.47,-5.04,187.44,0.47,-0.1,1.024,-4.5993"),
    doCheckCRSArgs = FALSE))
}

crsNZGD49 <- function() {
  return(crsNZMG())
}

crsNZGD00 <- function() {
# This comes from http://www.linz.govt.nz/data/linz-data-service/guides-and-documentation/using-lds-xyz-services-in-leaflet
# This is also known as EPSG:2193
    return(CRS(paste0("+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 ",
                    "+x_0=1600000 +y_0=10000000 +ellps=GRS80 ",
                    "+towgs84=0,0,0,0,0,0,0 +units=m +no_defs")))

}


# A note on the parameter strings for WGS84
# from http://lists.maptools.org/pipermail/proj/2009-February/004446.html
# Instead of  using
#   +proj=latlong +ellps=WGS84 +towgs84=0,0,0
# you can use
#   +proj=latlong +datum=WGS84
# which is equivalent.

crsWGS84 <- function() {
  return(CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs", doCheckCRSArgs = FALSE)) #A
}