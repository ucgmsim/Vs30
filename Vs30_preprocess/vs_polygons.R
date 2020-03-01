 
library(sp)
rm(list=ls())
# must be run within this script directory because of these silly source paths
# script dir isn't helpful when running interactively
# script_dir = dirname(sys.frame(1)$ofile)
source("Kevin/loadVs.R")

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

polys = vs_NZGD49 %over% poly_NZGD49
df = data.frame("index"=polys$INDEX,
                "easting"=vs_NZGD49@coords[,1],
                "northing"=vs_NZGD49@coords[,2]
)
rownames(df) = c()
write.csv(df, "../Vs30_data/vs_index.csv")
