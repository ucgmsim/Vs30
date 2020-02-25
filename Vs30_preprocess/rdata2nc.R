
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
