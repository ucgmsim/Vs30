library(raster) # raster
library(rgdal) # shapefiles
library(rgeos) # gDistance

source("config.R")
source("Kevin/const.R")
source("Kevin/vspr.R")


# working files (slp in NZMG)
slp_nzsi_9c = as(raster(paste0(PREFIX, "slp_nzsi_9c.nc")), "SpatialGridDataFrame")
slp_nzni_9c = as(raster(paste0(PREFIX, "slp_nzni_9c.nc")), "SpatialGridDataFrame")
variogram_aak = read.csv(paste0("data/variogram_", GEOLOGY, "_v6.csv"))[2:10]
variogram_yca = read.csv(paste0("data/variogram_", TERRAIN, "_v7.csv"))[2:10]
class(variogram_aak) = class(variogram_yca) = c("variogramModel", "data.frame")

# lowest LINZ resolution 1:500k
# coast_poly to determine if on land or water, coast_line for distances
coast_poly = readOGR(dsn=paste0(PLOTRES, "Paths/lds-nz-coastlines-and-islands/EPSG_2193"), layer="nz-coastlines-and-islands-polygons-topo-1500k")
crs(coast_poly) = NZTM
coast_line = as(coast_poly, "SpatialLinesDataFrame")
coast_distance = function(xy, km=T) {
  # xy: SpatialPoints with CRS EPSG:2193 NZGD2000/NZTM
  nxy = length(xy)
  result = rep(0.0, nxy)
  mask = which(!is.na(over(xy, coast_poly)$name))
  result[mask] = apply(gDistance(xy[mask,], coast_line, byid=T), 2, min)
  if (km) return(result/1000.0)
  return(result)
}

# vs site properties
vspr = vspr_run()
# remove Q3 quality unless station name is 3 chars long.
vspr = vspr[(vspr$QualityFlag != "Q3" | nchar(as(vspr$StationID, "character")) == 3 | is.na(vspr$QualityFlag)),]
# remove points where MODEL predictions don't exist
vspr_aak = vspr[(!is.na(vspr[[paste0("Vs30_", GEOLOGY)]])),]
vspr_yca = vspr[(!is.na(vspr[[paste0("Vs30_", TERRAIN)]])),]
rm(vspr)


geology_model_run = function(model, only_id=F) {
  library(raster) # raster
  library(rgeos) # gDistance
  
  source(paste0("Kevin/model_", GEOLOGY, ".R"))
  
  xy00 = SpatialPoints(model[, c("x", "y")])
  crs(xy00) = NZTM
  model$aak_vs30 = NA
  model$aak_stdev = NA
  
  # large amount of memory for polygon dataset
  gid_aak = over(xy00, aak_map)$groupID_AhdiAK
  if (only_id) return(gid_aak)
  # used to intersect with where gid_aak is water, water replaced with NA
  # TODO: maybe just delete water polygons making it faster?
  valid_idx = which(!is.na(gid_aak))
  if (length(valid_idx) == 0) return(model)
  xy00 = xy00[valid_idx]
  gid_aak = gid_aak[valid_idx]
  
  # coast and slope only required if using hybrid model
  # coastline distance (not much memory)
  coastkm = coast_distance(xy00)
  # slope (more memory required for rasters)
  xy49 = spTransform(xy00, NZMG)
  slp09c = xy49 %over% slp_nzsi_9c
  slp09c[is.na(slp09c)] = (xy49 %over% slp_nzni_9c)[is.na(slp09c)]
  slp09c[is.na(slp09c)] = 0.0
  rm(xy00, xy49)
  
  model_params = data.frame(gid_aak, slp09c, coastkm)
  rm(gid_aak, slp09c, coastkm)
  names(model_params) = c("gid", "slp09c", "coastkm")
  
  # run model
  model$aak_vs30[valid_idx] = model_ahdiak_get_vs30(model_params, g06mod=T, g13mod=T)
  model$aak_stdev[valid_idx] = model_ahdiak_hybrid$stdv(model_params$gid)
  
  return(model)
}


terrain_model_run = function(model, only_id=F) {
  library(raster) # crs, SpatialPoints
  
  source(paste0("Kevin/model_", TERRAIN, ".R"))
  
  xy00 = SpatialPoints(model[, c("x", "y")])
  crs(xy00) = NZTM
  model$yca_vs30 = NA
  model$yca_stdev = NA
  
  gid_yca = over(xy00, iwahashipike)$IwahashiPike_NZ_100m_16
  if (only_id) return(gid_yca)
  valid_idx = which(!is.na(gid_yca))
  if (length(valid_idx) == 0) return(model)

  # run model
  model$yca_vs30[valid_idx] = model_yongca$vs30[gid_yca]
  model$yca_stdev[valid_idx] = model_yongca$stdv[gid_yca]
  
  return(model)
}


mvn_run = function(model, vspr, variogram, model_type) {
  # TODO: keep_original flag to not overwrite pre-mvn data
  library(gstat)
  library(Matrix)
  library(raster) # crs

  source("Kevin/mvn.R")
  
  if (model_type == "aak") {
      m_name = GEOLOGY
  } else if (model_type == "yca") {
      m_name = TERRAIN
  }
  m_vs30 = paste0(model_type, "_vs30")
  m_stdev = paste0(model_type, "_stdev")
  m_vs30_out = paste0(model_type, "_mvn_vs30")
  m_stdev_out = paste0(model_type, "_mvn_stdev")
  
  valid_idx = which(!is.na(model[, m_vs30]))
  if (length(valid_idx) == 0) {
    names(model)[names(model) == m_vs30] = m_vs30_out
    names(model)[names(model) == m_stdev] = m_stdev_out
    return(model)
  }
  coords = coordinates(model[valid_idx, c("x", "y")])
  rownames(coords) = NULL
  
  obs_locations = coordinates(vspr)
  obs_values_log = log(vspr[[paste0("Vs30_", m_name)]])
  obs_variances = (vspr[[paste0("stDv_", m_name)]])^2
  obs_residuals = log(vspr$Vs30) - obs_values_log
  obs_stdev_log = vspr$lnMeasUncer
  values_log = log(model[valid_idx, m_vs30])
  variances = model[valid_idx, m_stdev]^2
  mvn_out = mvn(obs_locations, coords, variances, variogram,
                values_log, obs_variances, obs_residuals,
                covReducPar, obs_values_log, obs_stdev_log)
  # save memory, overwrite instead of add columns
  names(model)[names(model) == m_vs30] = m_vs30_out
  names(model)[names(model) == m_stdev] = m_stdev_out
  aak_resid = mvn_out$pred - values_log
  model[valid_idx, m_stdev_out] = sqrt(mvn_out$var)
  model[valid_idx, m_vs30_out] = exp(values_log) * exp(aak_resid)
  
  return(model)
}


weighting_run = function(model, stdev_weight=F, k=1) {
  model$vs30 = NA
  model$stdev = NA
  
  valid_idx = intersect(which(!is.na(model$aak_mvn_vs30)),
                        which(!is.na(model$yca_mvn_vs30)))
  if (length(valid_idx) == 0) return(model)

  log_a = log(model[valid_idx, "aak_mvn_vs30"])
  log_y = log(model[valid_idx, "yca_mvn_vs30"])
  if (stdev_weight) {
    m_a = (model[valid_idx, "aak_mvn_stdev"] ^ 2) ^ -k
    m_y = (model[valid_idx, "yca_mvn_stdev"] ^ 2) ^ -k
    w_a = m_a / (m_a + m_y)
    w_y = m_y / (m_a + m_y)
  } else {
    w_a = 0.5
    w_y = 0.5
  }
  log_ay = log_a * w_a + log_y * w_y
  model[valid_idx, "vs30"] = exp(log_ay)

  sig1sq = model[valid_idx, "aak_mvn_stdev"] ^ 2
  sig2sq = model[valid_idx, "yca_mvn_stdev"] ^ 2
  # Reference: https://en.wikipedia.org/wiki/Mixture_distribution#Moments
  model[valid_idx, "stdev"] = (w_a * ((log_a - log_ay) ^ 2 + sig1sq) + 
                               w_y * ((log_y - log_ay) ^ 2 + sig2sq)) ^ 0.5

  return(model)
}
