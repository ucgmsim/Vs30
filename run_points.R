#!/usr/bin/env Rscript

source("Kevin/main.R")

###
### CUSTOM POINTS VERSION
###

OUT = "vs30out.csv"

# pick a method to load stations

# manual list:
xy = data.frame(x=c(174.780278, 177), y=c(-41.300278, -37.983333))

# generated R file:
#source("Misc/validation_stations.R")

# ll file:
#xy = read.table("/nesi/project/nesi00213/StationInfo/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll")[,1:2]
#names(xy) = c("x", "y")

coordinates(xy) = ~ x + y
crs(xy) = WGS84
model = data.frame(spTransform(xy, NZTM))

# run all the steps
model = geology_model_run(model)
model = terrain_model_run(model)
model = mvn_run(model, vspr_aak, variogram_aak, "aak")
model = mvn_run(model, vspr_yca, variogram_yca, "yca")
model = weighting_run(model)
write.csv(model, OUT)
