
library(ncdf4)
rm(list=ls())
setwd("/home/vap30/VsMap/Rdata/")


convert2nc = function(rdata, raster) {
    # extents
    dx = (raster@extent@xmax - raster@extent@xmin) / raster@ncols
    xmin = raster@extent@xmin + 0.5 * dx
    xmax = raster@extent@xmax - 0.5 * dx
    xx = seq(xmin, xmax, dx)
    dy = (raster@extent@ymax - raster@extent@ymin) / raster@nrows
    ymin = raster@extent@ymin + 0.5 * dy
    ymax = raster@extent@ymax - 0.5 * dy
    yy = seq(ymax, ymin, -dy)
    # netCDF meta
    eastings = ncdim_def("easting","meters_east",as.single(xx)) 
    northings = ncdim_def("northing","meters_north",as.single(yy)) 
    fillvalue = NA
    values = ncvar_def("z","meters",list(eastings,northings),fillvalue,"value",prec="single")
    # netCDF create
    ncfname = paste0(substr(rdata, 1, nchar(rdata) - 5), "nc")
    if (file.exists(ncfname)) 
        file.remove(ncfname)
    ncout = nc_create(ncfname, values, force_v4=TRUE)
    # netCDF save
    ncvar_put(ncout,values,raster@data@values)
    # netCDF meta
    ncatt_put(ncout,"easting", "axis", "X")
    ncatt_put(ncout,"northing", "axis", "Y")
    ncatt_put(ncout, 0, "source", rdata)
    # finished
    nc_close(ncout)
}

###
### ELEVATION / SLOPE
###

rdata = "nzni_9c_DEM.Rdata"
load(rdata)
convert2nc(rdata, nzni_9c)

rdata = "nzni_30c_DEM.Rdata"
load(rdata)
convert2nc(rdata, nzni_30c)

rdata = "nzsi_9c_DEM.Rdata"
load(rdata)
convert2nc(rdata, nzsi_9c)

rdata = "nzsi_30c_DEM.Rdata"
load(rdata)
convert2nc(rdata, nzsi_30c)

rdata = "nzni_9c_slp.Rdata"
load(rdata)
convert2nc(rdata, slp_nzni_9c)

rdata = "nzni_30c_slp.Rdata"
load(rdata)
convert2nc(rdata, slp_nzni_30c)

rdata = "nzsi_9c_slp.Rdata"
load(rdata)
convert2nc(rdata, slp_nzsi_9c)

rdata = "nzsi_30c_slp.Rdata"
load(rdata)
convert2nc(rdata, slp_nzsi_30c)

###
### VARIOGRAM
###

load("variogram_AhdiAK_noQ3_hyb09c_v6.Rdata")
write.csv(variogram, "variogram_AhdiAK_noQ3_hyb09c_v6.csv")
load("variogram_YongCA_noQ3_v7.Rdata")
write.csv(variogram, "variogram_YongCA_noQ3_v7.csv")

###
### QMAP
###

setwd("/home/vap30/big_noDB/geo")

load("QMAP_Seamless_July13K_NZGD00.Rdata")

# shapefile maxlen = 10 chars
names(map_NZGD00@data)[9] = "TERRANE_EQ"
names(map_NZGD00@data)[10] = "SUPRGRP_EQ"
names(map_NZGD00@data)[11] = "GRP_EQ"
names(map_NZGD00@data)[12] = "SUBGRP_EQ"
names(map_NZGD00@data)[13] = "FORMATN_EQ"
names(map_NZGD00@data)[14] = "MEMBER_EQ"
names(map_NZGD00@data)[21] = "DESCRIPTN"
names(map_NZGD00@data)[22] = "ROCK_GRP"
names(map_NZGD00@data)[25] = "SIMPL_NAME"
names(map_NZGD00@data)[27] = "K_GRP_NAME"
names(map_NZGD00@data)[29] = "QMAP_NUMBR"
names(map_NZGD00@data)[31] = "Shape_Len"
names(map_NZGD00@data)[39] = "ID_testing"
names(map_NZGD00@data)[40] = "ID_AAK"
names(map_NZGD00@data)[41] = "ID_AAK_KA"
names(map_NZGD00@data)[42] = "ID_AAK_K3"
names(map_NZGD00@data)[43] = "ID_AAK_3"
names(map_NZGD00@data)[44] = "ID_AAK_3M"
names(map_NZGD00@data)[45] = "ID_AAK_KAH"
names(map_NZGD00@data)[46] = "ID_AAK_3_H"
names(map_NZGD00@data)[47] = "Vs30_testn"
names(map_NZGD00@data)[48] = "SD_testn"
names(map_NZGD00@data)[49] = "Vs30_AAK"
names(map_NZGD00@data)[50] = "SD_AAK"
names(map_NZGD00@data)[51] = "V30_AAK_KA"
names(map_NZGD00@data)[52] = "SD_AAK_KA"
names(map_NZGD00@data)[53] = "V30_AAK_K3"
names(map_NZGD00@data)[54] = "SD_AAK_K3"
names(map_NZGD00@data)[55] = "V30_AAK_3"
names(map_NZGD00@data)[56] = "SD_AAK_3"
names(map_NZGD00@data)[57] = "V30_AAK_3M"
names(map_NZGD00@data)[58] = "SD_AAK_3M"

# write in 2 coordinate systems while we are here
writeOGR(obj=map_NZGD00, dsn="QMAP_Seamless_July13K_NZGD00", layer="nztm", driver="ESRI Shapefile")
nzmg = CRS(paste0(
    "+proj=nzmg +lat_0=-41 +lon_0=173 +x_0=2510000 +y_0=6023150 ",
    "+units=m +no_defs +ellps=intl ",
    "+towgs84=59.47,-5.04,187.44,0.47,-0.1,1.024,-4.5993"),
    doCheckCRSArgs=FALSE)
map_NZGD49 = spTransform(map_NZGD00, CRS=nzmg)
writeOGR(obj=map_NZGD49, dsn="QMAP_Seamless_July13K_NZGD00", layer="nzmg", driver="ESRI Shapefile")
