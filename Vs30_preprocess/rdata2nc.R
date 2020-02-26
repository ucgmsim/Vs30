
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

###
### QMAP
###

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

writeOGR(obj=map_NZGD00, dsn="QMAP_Seamless_July13K_NZGD00", layer="data", driver="ESRI Shapefile")
