
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

# POINTS DATA
df = data.frame("index"=polys$INDEX,
                "easting"=vs_NZGD49@coords[,1],
                "northing"=vs_NZGD49@coords[,2],
                "slp09c"=slp09c$slope,
                "slp30c"=slp30c$slope,
                "gid_yongca"=groupID_YongCA,
                "v30_mvn_aak_n3_h9c"=Vs30_MVN_AhdiAK_noQ3_hyb09c,
                "v30_mvn_yca_n3"=Vs30_MVN_YongCA_noQ3,
                "sd_mvn_aak_n3_h9c"=sigma_MVN_AhdiAK_noQ3_hyb09c,
                "sd_mvn_yca_n3"=sigma_MVN_YongCA_noQ3
)
rownames(df) = c()
write.csv(df, "../Vs30_data/vs_index.csv")
