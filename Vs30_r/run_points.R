source("run_grid.R")

###
### CUSTOM POINTS VERSION
###

# pick a method to load stations
xy = data.frame(x=c(174.780278, 177), y=c(-41.300278, -37.983333))
#source("Misc/validation_stations.R")
#xy = SpatialPoints(read.table("/nesi/project/nesi00213/StationInfo/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll"))

coordinates(xy) = ~ x + y
crs(xy) = WGS84
model = data.frame(spTransform(xy, NZTM))

# run all the steps
model = geology_model_run(model)
model = terrain_model_run(model)
model = mvn_run(model, vspr_aak, variogram_aak, "aak")
model = mvn_run(model, vspr_yca, variogram_yca, "yca")
model = weighting_run(model)
write.csv(model, paste0(OUT, "/", "custom_points.csv"))
