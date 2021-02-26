# silence annoying warnings (before importing rgdal)
options("rgdal_show_exportToProj4_warnings"="none")

library(raster) # raster
library(rgdal) # shapefiles
library(rgeos) # gDistance

source("config.R")
source("func/const.R")
source("func/vspr.R")


# working files (slp in NZMG)
slp_nzsi_9c = as(raster(paste0(PREFIX, "slp_nzsi_9c.nc")), "SpatialGridDataFrame")
slp_nzni_9c = as(raster(paste0(PREFIX, "slp_nzni_9c.nc")), "SpatialGridDataFrame")
variogram_aak = read.csv(paste0("data/variogram_", GEOLOGY, "_v6.csv"))[2:10]
variogram_yca = read.csv(paste0("data/variogram_", TERRAIN, "_v7.csv"))[2:10]
class(variogram_aak) = class(variogram_yca) = c("variogramModel", "data.frame")

# lowest LINZ resolution 1:500k
# coast_poly to determine if on land or water, coast_line for distances
coast_poly = readOGR(dsn=paste0(PREFIX, "/coast"), layer="nz-coastlines-and-islands-polygons-topo-1500k")
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

# vs observed, also creates posterior model
vspr = vspr_run(posterior_update=POSTERIOR_UPDATE, clusters=POSTERIOR_CLUSTERS, cpt=CPT)
# remove points where MODEL predictions don't exist
vspr_aak = vspr[(!is.na(vspr[["gid_aak"]])),]
vspr_yca = vspr[(!is.na(vspr[["gid_yca"]])),]
rm(vspr)


geology_model_run = function(model, only_id=F) {
    library(raster) # raster
    library(rgeos) # gDistance

    source(paste0("func/model_", GEOLOGY, ".R"))

    # find group IDs
    xy00 = SpatialPoints(model[, c("x", "y")], proj4string=crs(NZTM))
    gid_aak = model_ahdiak_get_gid(xy00)
    if (only_id) return(gid_aak)
    model$aak_vs30 = NA
    model$aak_stdev = NA
    # water or outside any polygons is NA
    valid_idx = which(!is.na(gid_aak))
    if (length(valid_idx) == 0) return(model)
    xy00 = xy00[valid_idx]
    model_params = data.frame(gid=gid_aak[valid_idx])
    rm(gid_aak); gc()

    # coast and slope required for hybrid model
    # coastline distance (not much memory)
    model_params$coastkm = coast_distance(xy00)
    # slope (more memory required for rasters)
    xy49 = spTransform(xy00, NZMG)
    # TODO: combine ni and si
    slp09c = xy49 %over% slp_nzsi_9c
    slp09c$slope[is.na(slp09c)] = (xy49[is.na(slp09c)] %over% slp_nzni_9c)$slope
    slp09c$slope[is.na(slp09c)] = 0.0
    model_params$slp09c = slp09c$slope
    rm(xy00, xy49, slp09c); gc()

    # run model
    model$aak_vs30[valid_idx] = model_ahdiak_get_vs30(model_params, g06mod=T, g13mod=T)
    model$aak_stdev[valid_idx] = model_ahdiak_hybrid$stdv[model_params$gid]

    return(model)
}


terrain_model_run = function(model, only_id=F) {
    library(raster) # crs, SpatialPoints

    xy00 = SpatialPoints(model[, c("x", "y")], crs(NZTM))
    gid_yca = model_yongca_get_gid(xy00)
    if (only_id) return(gid_yca)
    model$yca_vs30 = NA
    model$yca_stdev = NA
    valid_idx = which(!is.na(gid_yca))
    if (length(valid_idx) == 0) return(model)

    # run model
    model$yca_vs30[valid_idx] = model_yongca$vs30[gid_yca[valid_idx]]
    model$yca_stdev[valid_idx] = model_yongca$stdv[gid_yca[valid_idx]]

    return(model)
}


# run models for vspr
vspr_aak = geology_model_run(vspr_aak)
names(vspr_aak)[names(vspr_aak) == "aak_vs30"] = "model_vs30"
names(vspr_aak)[names(vspr_aak) == "aak_stdev"] = "model_stdv"
vspr_yca = terrain_model_run(vspr_yca)
names(vspr_yca)[names(vspr_yca) == "yca_vs30"] = "model_vs30"
names(vspr_yca)[names(vspr_yca) == "yca_stdev"] = "model_stdv"


mvn_run = function(model, vspr, variogram, model_type) {
    # TODO: keep_original flag to not overwrite pre-mvn data
    library(gstat)
    library(Matrix)
    library(raster) # crs

    source("func/mvn.R")

    m_vs30 = paste0(model_type, "_vs30")
    m_stdev = paste0(model_type, "_stdev")
    m_vs30_out = paste0(model_type, "_mvn_vs30")
    m_stdev_out = paste0(model_type, "_mvn_stdev")
    if (MVN_OVERWRITE) {
        # save memory, overwrite instead of add columns
        names(model)[names(model) == m_vs30] = m_vs30_out
        names(model)[names(model) == m_stdev] = m_stdev_out
        m_vs30 = m_vs30_out
        m_stdev = m_stdev_out
    } else {
        model[m_vs30_out] = NA
        model[m_stdev_out] = NA
    }

    valid_idx = which(!is.na(model[, m_vs30]))
    if (length(valid_idx) == 0) return(model)

    obs_model_values_log = log(vspr[["model_vs30"]])
    model_values_log = log(model[valid_idx, m_vs30])
    mvn_out = mvn(vspr[c("x", "y")],
                model[valid_idx, c("x", "y")],
                model[valid_idx, m_stdev]^2,
                variogram,
                model_values_log,
                vspr[["model_stdv"]] ^ 2,
                log(vspr$Vs30) - obs_model_values_log,
                covReducPar,
                obs_model_values_log,
                vspr$lnMeasUncer)

    model[valid_idx, m_stdev_out] = sqrt(mvn_out$var)
    model[valid_idx, m_vs30_out] = exp(model_values_log) *
                                   exp(mvn_out$pred - model_values_log)

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

gc()
