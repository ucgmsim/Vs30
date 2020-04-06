
# Gathers metadata for points with measured Vs30.
# only columns needed are coordinates, Vs30/lnMeasUncer, Vs30/stDV for model(s),
#                         QualityFlag/StationID for subsetting

library(sp)

source("shared.R")
source("Kevin/load_vs.R")

OUT = "../Vs30_data/vspr.csv"

vspr_run = function() {
  # source coordinates with metadata
  vs_NZGD49 = load_vs(downsample_McGann=TRUE)
  vs_NZGD00 = sp::spTransform(vs_NZGD49, NZTM)

  # remove points in the same location with the same Vs30
  mask = rep(TRUE, length(vs_NZGD00))
  dup_pairs = sp::zerodist(vs_NZGD00)
  for (i in seq(dim(dup_pairs)[1])) {
      if(vs_NZGD00[dup_pairs[i,1],]$Vs30 == vs_NZGD00[dup_pairs[i,2],]$Vs30) {
          mask[dup_pairs[i,2]] = FALSE
      }
  }
  vs_NZGD00 = vs_NZGD00[mask,]
  rm(vs_NZGD49)

  # create output table
  vspr = data.frame(coordinates(vs_NZGD00))
  names(vspr) = c("x", "y")

  # copy vs info
  vspr$Vs30 = vs_NZGD00$Vs30
  vspr$lnMeasUncer = vs_NZGD00$lnMeasUncer
  vspr$QualityFlag = vs_NZGD00$QualityFlag
  vspr$StationID = vs_NZGD00$StationID

  # add model values
  vspr = geology_model_run(vspr)
  names(vspr)[names(vspr) == "aak_vs30"] = paste0("Vs30_", GEOLOGY)
  names(vspr)[names(vspr) == "aak_stdev"] = paste0("stDv_", GEOLOGY)
  vspr = terrain_model_run(vspr)
  names(vspr)[names(vspr) == "yca_vs30"] = paste0("Vs30_", TERRAIN)
  names(vspr)[names(vspr) == "yca_stdev"] = paste0("stDv_", TERRAIN)

  write.csv(vspr, "../Vs30_data/vspr.csv")
}

vspr_run()
