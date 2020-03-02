#!/usr/bin/env python
"""
prepares data. load Vs30 data and find geology & topography for those locations.
load Vs30 data sources
find slope at each Vs30 location
find geologic unit at each Vs30 location
compute geologic average Vs30 values by category and compute residuals
converting Vs30 coordinates to NZGD2000 from NZMG
apply geology-based Vs30 functions to each point
find Iwahashi & Pike (IP) geology category for each point (input to Yong CA model, name of variables = groupID_YongCA, etc.)
save the Yong (2012) Vs30 value corresponding to each point (YongCA_2012)

OUTPUT:
vspr.???

INPUT:
nz[si|ni]_[9|30]c_[slp|DEM].nc
"""

import os

import geopandas
from h5py import File as h5open
import numpy as np
import pandas as pd
from pyproj import Proj, transform
from scipy.spatial import distance_matrix
from shapely.geometry import Point, Polygon

from load_vs import load_vs
#import models

# DEM/slope directory
DS_DIR = "/home/vap30/VsMap/Rdata"

vspr_file = "vspr.???"

# polygons
map_NZGD49 = geopandas.read_file("/home/vap30/big_noDB/geo/QMAP_Seamless_July13K_NZGD00/nzmg.shp")
geo_polygons = np.array(map_NZGD49["geometry"].array)
# coordinate conversion if input is in nztm
# requires pyproj 6
#nzmg = Proj("epsg:27200")
#nztm = Proj("epsg:2193")
#def tm2mg(p):
#    easts, norths = p.exterior.coords.xy
#    x, y = transform(nztm, nzmg, norths, easts)
#    xy = Polygon(list(zip(x, y)))
#    return xy
#tm2mg_vectorized = np.vectorize(tm2mg)
#geo_polygons = tm2mg_vectorized(geo_polygons)

# DEM/slope
si_9c_slp = h5open(os.path.join(DS_DIR, "nzsi_9c_slp.nc"))
si_30c_slp = h5open(os.path.join(DS_DIR, "nzsi_30c_slp.nc"))
si_9c_dem = h5open(os.path.join(DS_DIR, "nzsi_9c_DEM.nc"))
si_30c_dem = h5open(os.path.join(DS_DIR, "nzsi_30c_DEM.nc"))
ni_9c_slp = h5open(os.path.join(DS_DIR, "nzni_9c_slp.nc"))
ni_30c_slp = h5open(os.path.join(DS_DIR, "nzni_30c_slp.nc"))
ni_9c_dem = h5open(os.path.join(DS_DIR, "nzni_9c_DEM.nc"))
ni_30c_dem = h5open(os.path.join(DS_DIR, "nzni_30c_DEM.nc"))

# stored in vspr data
vspoints_NZGD49 = load_vs(downsample_mcgann=True)
# sort/categorize Vs data in terms of map polygons & geology metadata
geo_points = pd.Series(list(zip(vspoints_NZGD49.Easting, vspoints_NZGD49.Northing))).apply(lambda x: Point(x[0], x[1])).values

###
### POINT IN WHICH POLYGON TEST IN PYTHON (SLOW)
###
#def contains(a_polygon, a_point):
#    return a_polygon.contains(a_point)
#contains_vectorized = np.vectorize(contains)
#
# INDEX column in R is 1 indexed location in original polygons (equivalent to polys here)
# this excludes nan entries where outside all polygons, if using argmax with axis=1, get 0 instead of nan
#polys = np.argwhere(contains_vectorized(geo_polygons, geo_points[:, np.newaxis]))[:, 1]
# R version does not include polygon shapes for results but includes locations from point results
# maybe should replace polygons with points
#vspr = map_NZGD49.iloc[polys]
###
### POINT IN WHICH POLYGON: taken from R (cannot change load_vs ordering)
###
# floats even for index because np.nan for no matching polygon
# index, easting, northing, slp09c, slp30c
site_data = pd.read_csv("../Vs30_data/vs_index.csv", usecols=[1, 4, 5])
polys = site_data["index"].values
vspr = map_NZGD49(iloc[polys[np.invert(np.isnan(polys))].astype(int)])


"""
vspr <- spCbind(vspr,df)
rm(df, df_ni, df_si,
   na_ni_30c,
   na_si_30c,
   na_ni_09c,
   na_si_09c)


########################################################################################
########################################################################################
# This is where I assign the Iwahashi & Pike terrain categories.
# (note, for geology categories this is done by assigning new fields beginning with
# "groupID_" in the script classifyThings.R, but I need to do it here because classifyThings
# uses only polygon data. Here I need a pixel-based overlay similar to the slope and DEM
# overlays done above.)

IP <- raster("~/big_noDB/topo/terrainCats/IwahashiPike_NZ_100m_16.tif")
IPdf <- as(IP, "SpatialGridDataFrame")
source("R/IP_levels.R")
vsprNZGD00 <- convert2NZGD00(vspr)
groupID_YongCA <- vsprNZGD00 %over% IPdf
colnames(groupID_YongCA) <- "ID" # rename column for plyr::join
groupID_YongCA_names <- plyr::join(groupID_YongCA,IPlevels[[1]]) # should get message "joining by: ID"
vspr$groupID_YongCA <- vspr$groupID_YongCA_noQ3 <- groupID_YongCA_names$category



# 2018-11  -  I want to generate post-MVN residuals. This is done here. (Must be run again after MVN rasters have been created.)

MVN_geology <- raster("~/big_noDB/models/MVN_Vs30_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif")
MVN_terrain <- raster("~/big_noDB/models/MVN_Vs30_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif")
MVN_geology_sigma <- raster("~/big_noDB/models/MVN_stDv_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif")
MVN_terrain_sigma <- raster("~/big_noDB/models/MVN_stDv_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif")

MVNdf_geo <- as(MVN_geology, "SpatialGridDataFrame")
MVNdf_ter <- as(MVN_terrain, "SpatialGridDataFrame")
MVNdf_geo_sigma <- as(MVN_geology_sigma, "SpatialGridDataFrame")
MVNdf_ter_sigma <- as(MVN_terrain_sigma, "SpatialGridDataFrame")

Vs30_MVN_AhdiAK_noQ3_hyb09c <- vsprNZGD00 %over% MVNdf_geo
Vs30_MVN_YongCA_noQ3        <- vsprNZGD00 %over% MVNdf_ter
sigma_MVN_AhdiAK_noQ3_hyb09c <- vsprNZGD00 %over% MVNdf_geo_sigma
sigma_MVN_YongCA_noQ3        <- vsprNZGD00 %over% MVNdf_ter_sigma

names(Vs30_MVN_AhdiAK_noQ3_hyb09c) <- "Vs30"
names(Vs30_MVN_YongCA_noQ3)        <- "Vs30"
names(sigma_MVN_AhdiAK_noQ3_hyb09c) <- "sigma"
names(sigma_MVN_YongCA_noQ3)        <- "sigma"

vspr$Vs30_MVN_AhdiAK_noQ3_hyb09c <- Vs30_MVN_AhdiAK_noQ3_hyb09c$Vs30
vspr$Vs30_MVN_YongCA_noQ3        <- Vs30_MVN_YongCA_noQ3$Vs30
vspr$stDv_MVN_AhdiAK_noQ3_hyb09c <- sigma_MVN_AhdiAK_noQ3_hyb09c$sigma
vspr$stDv_MVN_YongCA_noQ3        <- sigma_MVN_YongCA_noQ3$sigma



# Here is where I create groupID_AhdiYongWeighted1 and similar. (This will produce an error
# if a new weighted model is added and not handled below. Update MANUALLY.)
for(weightedModelName in wtdMODELs) {
  # weightedModelName =  wtdMODELs[1] # testing
  if(identical(weightedModelName,  "AhdiYongWeighted1")) {
    vspr$groupID_AhdiYongWeighted1 <- interaction(vspr$groupID_AhdiAK,  vspr$groupID_YongCA)
  } else {stop("Weighted model is not currently handled in vspr.R")}
}



############
####################################
#################################################################
# The following needs to be updated MANUALLY when new models are produced.
#################################################################
####################################
############
# Produce model estimates for points based on slope criteria
# (geology-only models are already handled by processQmap.R)

# Vs30_AhdiAK_KaiAll_hyb09c <- AhdiAK_KaiAll_hyb09c_set_Vs30(as.data.frame(vspr))  # KaiAll model has been removed.
# stDv_AhdiAK_KaiAll_hyb09c <- AhdiAK_KaiAll_hyb09c_set_stDv(as.data.frame(vspr))  # KaiAll model has been removed.

Vs30_AhdiAK_noQ3_hyb09c <- AhdiAK_noQ3_hyb09c_set_Vs30(as.data.frame(vspr))     # 20171205
stDv_AhdiAK_noQ3_hyb09c <- AhdiAK_noQ3_hyb09c_set_stDv(as.data.frame(vspr))     # 20171205

Vs30_YongCA             <- YongCA_set_Vs30(as.data.frame(vspr))                 # 20171218
stDv_YongCA             <- YongCA_set_stDv(as.data.frame(vspr))                 # 20171218

Vs30_YongCA_noQ3        <- YongCA_noQ3_set_Vs30(as.data.frame(vspr))            # 20171218
stDv_YongCA_noQ3        <- YongCA_noQ3_set_stDv(as.data.frame(vspr))            # 20171218

Vs30_AhdiYongWeighted1  <- exp(0.5 * (log(Vs30_AhdiAK_noQ3_hyb09c)   + log(Vs30_YongCA_noQ3)))  # 20180110
mu1 <- log(Vs30_AhdiAK_noQ3_hyb09c)
mu2 <- log(Vs30_YongCA_noQ3)
mu  <- log(Vs30_AhdiYongWeighted1)
sig1sq <- stDv_AhdiAK_noQ3_hyb09c^2
sig2sq <- stDv_YongCA_noQ3^2
sigsq  <- 0.5 * ( (( mu1 - mu)^2) + sig1sq +
                    (( mu2 - mu)^2) + sig2sq)
stDv_AhdiYongWeighted1  <- sqrt(sigsq)              # FIXED on 20180405


vspr <- spCbind(vspr, data.frame(
                                 # Vs30_AhdiAK_KaiAll_hyb09c,   # KaiAll model has been removed.
                                 # stDv_AhdiAK_KaiAll_hyb09c,   # KaiAll model has been removed.
                                 Vs30_AhdiAK_noQ3_hyb09c,                       # 20171205
                                 stDv_AhdiAK_noQ3_hyb09c,                       # 20171205
                                 Vs30_YongCA,                                   # 20171218
                                 stDv_YongCA,                                   # 20171218
                                 Vs30_YongCA_noQ3,                              # 20171218
                                 stDv_YongCA_noQ3,                              # 20171218
                                 Vs30_AhdiYongWeighted1,                        # 20180110
                                 stDv_AhdiYongWeighted1                         # 20180110
                                 ))


# Add Vs metadata
vspr <- spCbind(vspr,VsPts_NZGD49@data)
rm(VsPts_NZGD49)



# Convert to NZGD2000
source("~/VsMap/R/geodetic.R")
vspr <- convert2NZGD00(vspr)


# =============================================================================
# Cull vspr points based on removal criteria -----------------------------------------

# WATER points.....
# at some point may fix this by finding nearest value.
# possible ways to do this discussed here:
#  http://stackoverflow.com/questions/26308426/how-do-i-find-the-polygon-nearest-to-a-point-in-r
vspr$cull_water   <- (vspr$UNIT_CODE == "water")
vspr$cull_na      <- (is.na(vspr$UNIT_CODE))


# SLOPELESS points -------------
vspr$cull_noSlope09 <- is.na(vspr$slp09c)
vspr$cull_noSlope30 <- is.na(vspr$slp30c)
vspr$cull_noSlope   <- vspr$cull_noSlope09 | vspr$cull_noSlope30


# DUPLICATE points ------------
vspr$cull_duplicate     <- logical(dim(vspr)[1])


duplicatePoints <- zerodist(vspr)
dupeVs30        <- numeric(dim(duplicatePoints)[1]) # initialize vector
for (i in seq_along(dupeVs30)) {
  if(vspr[duplicatePoints[i,1],]$Vs30 ==  # if the two points have same Vs30 value... 
     vspr[duplicatePoints[i,2],]$Vs30) {
    dupeVs30[i] <- vspr[duplicatePoints[i,1],]$Vs30
    vspr$cull_duplicate[i] <- T  # mark datapoint for removal
  } else { # if the Vs30 values differ, retain both points.
    dupeVs30[i] <- 0 # mark this by 0
  }
}
removeTheseDupes <- dupeVs30>0
df <- data.frame(point1_idx=duplicatePoints[,1],
                 point2_idx=duplicatePoints[,2],
                 point1_Vs30=vspr[duplicatePoints[,1],]$Vs30,
                 point2_Vs30=vspr[duplicatePoints[,2],]$Vs30,
                 dupeVs30=dupeVs30,
                 remove=removeTheseDupes
                 ) # making this dataframe for reporting cull as txtfile

# write duplicate culling removal info to a text file
# (first, CSV in ./tmp/ , then call util-linux "column" command via system2)
write.csv(df,file = "tmp/duplicatePointsRemoval.csv",row.names = F)
system2("column",args=c("-t",
                        "-s",
                        ","),
                 stdin = "tmp/duplicatePointsRemoval.csv",
                 stdout= "out/duplicatePointsRemoval.csv")


# all rows marked as "cull"
vspr$cull         <- vspr$cull_noSlope | 
                     vspr$cull_water | 
                     vspr$cull_na |
                     vspr$cull_duplicate


vspr_pre_cull <- vspr # Save old version first - with info about which rows are marked to be culled.
vspr <- vspr[vspr$cull==FALSE,]







# =============================================================================
# Extract geology statistics and compute slope residuals -----------------
avgs <- aggregate.data.frame(x = vspr$Vs30, by = list(vspr$dep), FUN = mean)
stdv <- aggregate.data.frame(x = vspr$Vs30, by = list(vspr$dep), FUN = sd)
colnames(avgs) <- c("dep", "meanVsByGroup")
colnames(stdv) <- c("dep", "stDevVsByGroup")

vspr2 <- merge(vspr, avgs, by = "dep")
vspr3 <- merge(vspr2, stdv, by = "dep")
vspr <- vspr3
rm(vspr2, vspr3)


# rGeo is the geology residual - equivalent to that in Thompson et al 2014
vspr$rGeo_byMeans  <- vspr$Vs30 / vspr$meanVsByGroup



save(vspr_pre_cull,vspr,file = vspr_file) # Save "pre-cull vspr"














#==============================================================================
# Compute residuals
setwd("~/VsMap/")
load(vspr_file) 
library(sp)
source("R/functions.R")
source("R/models.R")



############
####################################
#################################################################
# The following needs to be updated MANUALLY when new models are produced.
#################################################################
####################################
############


res_testing               <- log(vspr$Vs30) - log(vspr$Vs30_testing)
res_AhdiAK                <- log(vspr$Vs30) - log(vspr$Vs30_AhdiAK)
res_AhdiAK_KaiAll         <- log(vspr$Vs30) - log(vspr$Vs30_AhdiAK_KaiAll)
res_AhdiAK_KaiNoQ3        <- log(vspr$Vs30) - log(vspr$Vs30_AhdiAK_KaiNoQ3)
# res_AhdiAK_KaiAll_hyb09c  <- log(vspr$Vs30) - log(vspr$Vs30_AhdiAK_KaiAll_hyb09c)    # KaiAll model has been removed.
res_AhdiAK_noQ3           <- log(vspr$Vs30) - log(vspr$Vs30_AhdiAK_noQ3)          # added 20171129
res_AhdiAK_noQ3noMcGann   <- log(vspr$Vs30) - log(vspr$Vs30_AhdiAK_noQ3noMcGann)  # added 20171129
res_AhdiAK_noQ3_hyb09c    <- log(vspr$Vs30) - log(vspr$Vs30_AhdiAK_noQ3_hyb09c)   # added 20171205
res_YongCA                <- log(vspr$Vs30) - log(vspr$Vs30_YongCA)               # added 20171218
res_YongCA_noQ3           <- log(vspr$Vs30) - log(vspr$Vs30_YongCA_noQ3)          # added 20171218
res_AhdiYongWeighted1     <- log(vspr$Vs30) - log(vspr$Vs30_AhdiYongWeighted1)    # added 20180110
res_MVN_AhdiAK_noQ3_hyb09c <- log(vspr$Vs30) - log(vspr$Vs30_MVN_AhdiAK_noQ3_hyb09c) # added 20181107
res_MVN_YongCA_noQ3        <- log(vspr$Vs30) - log(vspr$Vs30_MVN_YongCA_noQ3)        # added 20181107


vspr <- spCbind(vspr, data.frame(res_testing,
                                 res_AhdiAK,
                                 res_AhdiAK_KaiAll,
                                 res_AhdiAK_KaiNoQ3,
                                 # res_AhdiAK_KaiAll_hyb09c,    # KaiAll model has been removed.
                                 res_AhdiAK_noQ3,                                 # added 20171129
                                 res_AhdiAK_noQ3noMcGann,                         # added 20171129
                                 res_AhdiAK_noQ3_hyb09c,                          # added 20171205
                                 res_YongCA,                                      # added 20171218
                                 res_YongCA_noQ3,                                 # added 20171218
                                 res_AhdiYongWeighted1,                           # added 20180110
                                 res_MVN_AhdiAK_noQ3_hyb09c,                      # added 20181107
                                 res_MVN_YongCA_noQ3                              # added 20181107
                                 ))

save(vspr_pre_cull, vspr, file = vspr_file)




#################################################################################################
# output tables for GMT to use.
#
source("R/vspr_write.R")
vspr_write()

"""
