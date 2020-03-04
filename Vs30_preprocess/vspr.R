
# Gathers metadata for points with measured Vs30.

# must be run within this script directory because of these silly source paths
# script dir isn't helpful when running interactively
# script_dir = dirname(sys.frame(1)$ofile)
rm(list=ls())
library(sp)

# POINTS
source("Kevin/loadVs.R")
vs_NZGD49 = loadVs(downSampleMcGann=TRUE)
vs_NZGD00 = spTransform(vs_NZGD49, CRS(paste0("+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 ",
                                              "+x_0=1600000 +y_0=10000000 +ellps=GRS80 ",
                                              "+towgs84=0,0,0,0,0,0,0 +units=m +no_defs")))

# POLYGONS
load(file = "~/big_noDB/geo/QMAP_Seamless_July13K_NZGD00.Rdata")
poly_NZGD49 = spTransform(map_NZGD00, CRS=CRS(paste0(
    "+proj=nzmg +lat_0=-41 +lon_0=173 +x_0=2510000 +y_0=6023150 ",
    "+units=m +no_defs +ellps=intl ",
    "+towgs84=59.47,-5.04,187.44,0.47,-0.1,1.024,-4.5993"),
    doCheckCRSArgs=FALSE)
)

# POLYGONS for POINTS
polys = vs_NZGD49 %over% poly_NZGD49
rm(poly_NZGD49)

# SLOPE for POINTS
load(file="~/VsMap/Rdata/nzsi_9c_slp.Rdata")
load(file="~/VsMap/Rdata/nzsi_30c_slp.Rdata")
load(file="~/VsMap/Rdata/nzni_9c_slp.Rdata")
load(file="~/VsMap/Rdata/nzni_30c_slp.Rdata")
slp09c = vs_NZGD49 %over% slp_nzsi_9c.sgdf
slp30c = vs_NZGD49 %over% slp_nzsi_30c.sgdf
slp09c[is.na(slp09c)] = (vs_NZGD49 %over% slp_nzni_9c.sgdf)[is.na(slp09c)]
slp30c[is.na(slp30c)] = (vs_NZGD49 %over% slp_nzni_30c.sgdf)[is.na(slp30c)]
rm(slp_nzsi_9c, slp_nzsi_9c.sgdf, slp_nzsi_30c, slp_nzsi_30c.sgdf,
   slp_nzni_9c, slp_nzni_9c.sgdf, slp_nzni_30c, slp_nzni_30c.sgdf)

# YONG CATEGORIES for POINTS
IP = as(raster("~/big_noDB/topo/terrainCats/IwahashiPike_NZ_100m_16.tif"), "SpatialGridDataFrame")
# originally duplicated as groupID_YongCA and groupID_YoungCA_noQ3
# also the duplicated data wasn't the integer ID (16 groups) but the full text description
groupID_YongCA = vs_NZGD00 %over% IP
rm(IP)

# MVN for POINTS
MVN_geology = as(raster("~/big_noDB/models/MVN_Vs30_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif"), "SpatialGridDataFrame")
MVN_terrain = as(raster("~/big_noDB/models/MVN_Vs30_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif"), "SpatialGridDataFrame")
MVN_geology_sigma = as(raster("~/big_noDB/models/MVN_stDv_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif"), "SpatialGridDataFrame")
MVN_terrain_sigma = as(raster("~/big_noDB/models/MVN_stDv_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif"), "SpatialGridDataFrame")
Vs30_MVN_AhdiAK_noQ3_hyb09c = vs_NZGD00 %over% MVN_geology
Vs30_MVN_YongCA_noQ3 = vs_NZGD00 %over% MVN_terrain
sigma_MVN_AhdiAK_noQ3_hyb09c = vs_NZGD00 %over% MVN_geology_sigma
sigma_MVN_YongCA_noQ3 = vs_NZGD00 %over% MVN_terrain_sigma
rm(MVN_geology, MVN_terrain, MVN_geology_sigma, MVN_terrain_sigma)

# MODELS for POINTS
# model groups saved instead, should really be numeric IDs but are description categories instead
# AhdiAK_noQ3_hyb09c
# YongCA
# YongCA_noQ3
# AhdiYongWeighted1

# POINTS DATA
# add columns as required from vs_NZGD00 and polys
# groupid for YoungCA and YoungCA_noQ3 are equal so not duplicated
# groupid for AhdiYongWeighted1 is just a concatenation of Ahdi and Yong groups
df = data.frame("polygon"=polys$INDEX,
                "easting"=vs_NZGD00@coords[,1],
                "northing"=vs_NZGD00@coords[,2],
                "v30"=vs_NZGD00$Vs30,
                "terrain"=vspr$dep,
                "slp09c"=slp09c$slope,
                "slp30c"=slp30c$slope,
                "v30_mvn_aak_n3_h9c"=Vs30_MVN_AhdiAK_noQ3_hyb09c,
                "v30_mvn_yca_n3"=Vs30_MVN_YongCA_noQ3,
                "sd_mvn_aak_n3_h9c"=sigma_MVN_AhdiAK_noQ3_hyb09c,
                "sd_mvn_yca_n3"=sigma_MVN_YongCA_noQ3,
                "gid_yca"=groupID_YongCA,
                "gid_aak_n3_h9c"=polys$groupID_AhdiAK_noQ3_hyb09c
)
rownames(df) = c()

# POINTS CLEAN
# remove points in water, nans
# potentially do before finding metadata for them (along the way)
unwanted = (polys$UNIT_CODE == "water") |
           (is.na(polys$UNIT_CODE)) |
           (is.na(slp09c)) |
           (is.na(slp30c))
# also remove points in the same location with the same Vs30, Vs30 to be stored in vspr?
dup_pairs = zerodist(vs_NZGD00)
for (i in seq(dim(dup_pairs)[1])) {
    if(vs_NZGD00[dup_pairs[i,1],]$Vs30 == vs_NZGD00[dup_pairs[i,2],]$Vs30) {
        unwanted[dup_pairs[i,2]] = T
    }
}
df = df[unwanted==FALSE,]

# duplicate data and data with different dimensions:
# 1. residuals will not be saved in vspr table
#    they are simply log(vs30) - log(other vs30)
# 2. mean and sd of vs30 by terrain category, or these can be saved in another table
# 3. rGeos: vs30 / mean of vs30 by terrain category

write.csv(df, "../Vs30_data/vspr.csv")
