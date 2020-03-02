 
library(sp)
rm(list=ls())
# must be run within this script directory because of these silly source paths
# script dir isn't helpful when running interactively
# script_dir = dirname(sys.frame(1)$ofile)
source("Kevin/loadVs.R")

load(file="~/VsMap/Rdata/nzsi_9c_slp.Rdata")
load(file="~/VsMap/Rdata/nzsi_30c_slp.Rdata")
load(file="~/VsMap/Rdata/nzni_9c_slp.Rdata")
load(file="~/VsMap/Rdata/nzni_30c_slp.Rdata")

# POINTS
vs_NZGD49 = loadVs(downSampleMcGann=TRUE)

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

# SLOPE for POINTS
slp09c = vs_NZGD49 %over% slp_nzsi_9c.sgdf
slp30c = vs_NZGD49 %over% slp_nzsi_30c.sgdf
slp09c[is.na(slp09c)] = (vs_NZGD49 %over% slp_nzni_9c.sgdf)[is.na(slp09c)]
slp30c[is.na(slp30c)] = (vs_NZGD49 %over% slp_nzni_30c.sgdf)[is.na(slp30c)]

# POINTS DATA
df = data.frame("index"=polys$INDEX,
                "easting"=vs_NZGD49@coords[,1],
                "northing"=vs_NZGD49@coords[,2],
                "slp09c"=slp09c$slope,
                "slp30c"=slp30c$slope
)
rownames(df) = c()
write.csv(df, "../Vs30_data/vs_index.csv")
