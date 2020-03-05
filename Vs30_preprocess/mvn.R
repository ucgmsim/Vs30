
library(gstat)
library(Matrix)
library(raster)

source("R/mvn_params.R")
source("R/mvnRaster.R")

# vs site properties
load("Rdata/vspr.Rdata")
vspr_noQ3 = vspr[(vspr$QualityFlag != "Q3" | is.na(vspr$QualityFlag)),]

# remove points where MODEL predictions don't exist
model_na = which(is.na(vspr_noQ3[[paste0("Vs30_", MODEL)]]))
if(length(model_na) > 0) {
  warning("some locations don't have predictions for this model")
  vspr_noQ3 = vspr_noQ3[-model_na,]
}

# import variogram
load(sprintf("Rdata/variogram_%s_%s.Rdata", MODEL, vgName))


mvn_oneTile = function(x, y, vspr, variogram, MODEL) {
  tile_name = sprintf("x%02dy%02d", x, y)
  load(sprintf("Rdata/INDEX_NZGD00_allNZ_%s.Rdata", tile_name))
  exts = extent(indexRaster)
  
  # this should be outside the loop
  vs30 = raster(switch(MODEL,
        AhdiAK             = "~/big_noDB/models/geo_NZGD00_allNZ_AhdiAK.tif",
        AhdiAK_noQ3        = "~/big_noDB/models/geo_NZGD00_allNZ_AhdiAK_noQ3.tif",
        AhdiAK_noQ3_hyb09c = "~/big_noDB/models/hyb_NZGD00_allNZ_AhdiAK_noQ3_hyb09c.tif",
        YongCA             = "~/big_noDB/models/Yong2012_Cali_Vs30.tif",
        YongCA_noQ3        = "~/big_noDB/models/YongCA_noQ3.tif",
        AhdiYongWeighted1  = "~/big_noDB/models/AhdiYongWeighted1.tif"))       
  stdev = raster(switch(MODEL,
        AhdiAK             = "~/big_noDB/models/sig_NZGD00_allNZ_AhdiAK.tif",
        AhdiAK_noQ3        = "~/big_noDB/models/sig_NZGD00_allNZ_AhdiAK_noQ3.tif",
        AhdiAK_noQ3_hyb09c = "~/big_noDB/models/sig_NZGD00_allNZ_AhdiAK_noQ3_hyb09c.tif",
        YongCA             = "~/big_noDB/models/YongCA_sigma.tif",
        YongCA_noQ3        = "~/big_noDB/models/YongCA_noQ3_sigma.tif",
        AhdiYongWeighted1  = "~/big_noDB/models/AhdiYongWeighted1_sigma.tif"))       
  vs30 = crop(vs30, exts)
  stdev = crop(stdev, exts)
  
  mvnOutput = mvnRaster(indexRaster, vs30, stdev, vspr, variogram, MODEL, covReducPar)
  mvnVs30 = vs30 * exp(mvnOutput$lnObsPred)

  save(mvnOutput, mvnVs30, 
       file=sprintf("Rdata/MVNRM_NZGD00_allNZ_%s_%s_noisy%s_minDist%03.1fkm_%s_crp%03.1f.Rdata",
                      tile_name, MODEL,
                      substr(as.character(useNoisyMeasurements),1,1),
                      minThreshold/1000,
                      vgName, covReducPar))
  
  for (prefix in c("MVkrg", "MVres", "MVsdv")) {
    rast = switch(prefix,
                   MVkrg = mvnVs30,
                   MVres = mvnOutput$lnObsPred,
                   MVsdv = mvnOutput$stdDev)
    writeRaster(rast, filename=paste0("tmp/", prefix, "_NZGD00_allNZ_", tile_name, "_", MODEL,
                                      "_noisy", substr(as.character(useNoisyMeasurements), 1, 1),
                                      sprintf("_minDist%03.1fkm", minThreshold/1000),
                                      "_", vgName, sprintf("_crp%03.1f", covReducPar),
                                      ".tif"), format="GTiff", overwrite=TRUE)
  }
}


x = 1
y = 1
mvn_oneTile(x, y, vspr_noQ3, variogram, MODEL)