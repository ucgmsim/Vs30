library("raster")

source("config.R")

# NZTM
yca_map = as(raster(paste0(PREFIX, "IwahashiPike_NZ_100m_16.tif")), "SpatialGridDataFrame")

model_yongca_get_gid = function(coords) {
    return(over(coords, yca_map)$IwahashiPike_NZ_100m_16)
}
