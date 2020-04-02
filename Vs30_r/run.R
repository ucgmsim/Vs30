library(parallel) # cluster
library(raster) # raster
library(rgdal) # shapefiles
library(rgeos) # gDistance

# outputs placed into this directory
OUT = "vs30out"
PLOTRES = "/nesi/project/nesi00213/PlottingData/"
PREFIX = "/nesi/project/nesi00213/PlottingData/Vs30/"

GEOLOGY = "AhdiAK_noQ3_hyb09c"
TERRAIN = "YongCA_noQ3"

WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
NZTM = "+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
NZMG = "+proj=nzmg +lat_0=-41 +lon_0=173 +x_0=2510000 +y_0=6023150 +units=m +no_defs +ellps=intl +towgs84=59.47,-5.04,187.44,0.47,-0.1,1.024,-4.5993"

# EX mvn_params.R
# note that some if statements have been removed to match default values
# should note down where this was and/or change back if ever modifying below values
# at least applicable to useNoisyMeasurements, note loadVs.R
numVGpoints = 128
useNoisyMeasurements = T
covReducPar = 1.5
useDiscreteVariogram = F
useDiscreteVariogram_replace = F
optimizeUsingMatrixPackage = T

# working files (aak_map, iwahashipike in NZTM, slp in NZMG)
load(paste0(PREFIX, "aak_map.Rdata"))
slp_nzsi_9c = as(raster(paste0(PREFIX, "slp_nzsi_9c.nc")), "SpatialGridDataFrame")
slp_nzni_9c = as(raster(paste0(PREFIX, "slp_nzni_9c.nc")), "SpatialGridDataFrame")
iwahashipike = as(raster(paste0(PREFIX, "IwahashiPike_NZ_100m_16.tif")), "SpatialGridDataFrame")
variogram_aak = read.csv(paste0("../Vs30_data/variogram_", GEOLOGY, "_v6.csv"))[2:10]
variogram_yca = read.csv(paste0("../Vs30_data/variogram_", TERRAIN, "_v7.csv"))[2:10]
class(variogram_aak) = class(variogram_yca) = c("variogramModel", "data.frame")

# lowest LINZ resolution 1:500k
# coast_poly to determine if on land or water, coast_line for distances
coast_poly = readOGR(dsn=paste0(PLOTRES, "Paths/lds-nz-coastlines-and-islands/EPSG_2193"), layer="nz-coastlines-and-islands-polygons-topo-1500k")
coast_line = as(coast_poly, "SpatialLinesDataFrame")
coast_distance = function(xy, km=T) {
  # xy: SpatialPoints with CRS EPSG:2193 NZGD2000/NZTM
  nxy = length(xy)
  result = rep(0.0, nxy)
  mask = which(!is.na(over(xy, coast_poly)$name))
  result[mask] = apply(gDistance(xy[mask,], coast_line, byid=T), 2, min)
  if (km) {return(result/1000.0)}
  return(result)
}

# vs site properties
# TODO: cleanup vspr, full of duplicate and unused columns
load("~/VsMap/Rdata/vspr.Rdata")
rm(vspr_pre_cull)
# remove Q3 quality unless station name is 3 chars long.
# TODO: just call loadVs function and it will do it for you instead of here
vspr = vspr[(vspr$QualityFlag != "Q3" | nchar(vspr$StationID) == 3 | is.na(vspr$QualityFlag)),]

# remove points where MODEL predictions don't exist, helper function within loadVs?
na = which(is.na(vspr[[paste0("Vs30_", GEOLOGY)]]))
if(length(na) > 0) {
  warning("some measured locations have no value for AhdiAK")
  vspr_aak = vspr[-na,]
} else {
  vspr_aak = vspr
}
na = which(is.na(vspr[[paste0("Vs30_", TERRAIN)]]))
if(length(na) > 0) {
  warning("some measured locations have no value for YongCA")
  vspr_yca = vspr[-na,]
} else {
  vspr_yca = vspr
}
rm(vspr, na)
# update model values with geology model tweaks
# TODO: use function instead to generate freshly, this step won't be necessary
source(paste0("Kevin/MODEL_", GEOLOGY, ".R"))
vspr_aak_points = SpatialPoints(vspr_aak@coords, proj4string=vspr_aak@proj4string)
distances = coast_distance(vspr_aak_points)
xy49 = spTransform(vspr_aak_points, NZMG)
slp09c = xy49 %over% slp_nzsi_9c
slp09c[is.na(slp09c)] = (xy49 %over% slp_nzni_9c)[is.na(slp09c)]
slp09c[is.na(slp09c)] = 0.0
model_params = data.frame(vspr_aak@data$groupID_AhdiAK, slp09c, distances)
names(model_params) = c("groupID_AhdiAK", "slp09c", "coastkm")
vspr_aak[[paste0("Vs30_", GEOLOGY)]] = AhdiAK_noQ3_hyb09c_set_Vs30(model_params, g06mod=T, g13mod=T)
rm(model_params, slp09c, distances, vspr_aak_points, xy49)


geology_model_run = function(model) {
  library(raster) # raster
  library(rgeos) # gDistance
  
  source("Kevin/MODEL_AhdiAK_noQ3_hyb09c.R")
  
  xy00 = SpatialPoints(model[, c("x", "y")])
  crs(xy00) = NZTM
  model$aak_values_log = NA
  model$aak_variances = NA
  
  # large amount of memory for polygon dataset
  gid_aak = over(xy00, aak_map)$groupID_AhdiAK
  # make sure datatype is correct
  # TODO: short integer based model to save memory (use int indexes instead of string)?
  if (length(xy00) > 1) {
    gid_aak = as.character(gid_aak)
  } else {
    gid_aak = lapply(gid_aak, as.character)
  }
  valid_idx = intersect(which(!is.na(gid_aak)), which(gid_aak != "00_WATER"))
  if (length(valid_idx) == 0) {return(model)}
  xy00 = xy00[valid_idx]
  gid_aak = gid_aak[valid_idx]
  
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
  names(model_params) = c("groupID_AhdiAK", "slp09c", "coastkm")
  
  # run model
  model$aak_values_log[valid_idx] = log(AhdiAK_noQ3_hyb09c_set_Vs30(model_params, g06mod=T, g13mod=T))
  model$aak_variances[valid_idx] = AhdiAK_noQ3_hyb09c_set_stDv(model_params)^2
  
  return(model)
}


terrain_model_run = function(model) {
  library(raster) # crs, SpatialPoints
  
  source("Kevin/MODEL_YongCA_noQ3.R")
  
  xy00 = SpatialPoints(model[, c("x", "y")])
  crs(xy00) = NZTM
  model$yca_values_log = NA
  model$yca_variances = NA
  
  gid_yca = over(xy00, iwahashipike)
  valid_idx = which(!is.na(gid_yca))
  if (length(valid_idx) == 0) {return(model)}
  gid_yca = data.frame(gid_yca[valid_idx,])
  colnames(gid_yca) = "groupID_YongCA_noQ3"

  # run model
  model$yca_values_log[valid_idx] = log(YongCA_noQ3_set_Vs30(gid_yca))
  model$yca_variances[valid_idx] = YongCA_noQ3_set_stDv(gid_yca)^2
  
  return(model)
}


mvn_run = function(model, vspr, variogram, model_type) {
  library(gstat)
  library(Matrix)
  library(raster) # crs

  source("Kevin/mvn.R")
  
  if (model_type == "aak") {
      m_name = GEOLOGY
  } else if (model_type == "yca") {
      m_name = TERRAIN
  }
  m_values_log = paste0(model_type, "_values_log")
  m_variances = paste0(model_type, "_variances")
  m_vs30 = paste0(model_type, "_vs30")
  m_stdev = paste0(model_type, "_stdev")
  
  valid_idx = which(!is.na(model[, m_values_log]))
  if (length(valid_idx) == 0) {
    names(model)[names(model) == m_values_log] = m_vs30
    names(model)[names(model) == m_variances] = m_stdev
    return(model)
  }
  coords = coordinates(model[valid_idx, c("x", "y")])
  rownames(coords) = NULL
  
  obs_locations = coordinates(vspr)
  obs_values_log = log(vspr[[paste0("Vs30_", m_name)]])
  obs_variances = (vspr[[paste0("stDv_", m_name)]])^2
  obs_residuals = log(vspr$Vs30) - obs_values_log
  obs_stdev_log = vspr$lnMeasUncer
  values_log = model[valid_idx, m_values_log]
  mvn_out = mvn(obs_locations, coords, model[valid_idx, m_variances], variogram,
                values_log, obs_variances, obs_residuals,
                covReducPar, obs_values_log, obs_stdev_log)
  # save memory, overwrite instead of add columns
  names(model)[names(model) == m_values_log] = m_vs30
  names(model)[names(model) == m_variances] = m_stdev
  aak_resid = mvn_out$pred - values_log
  model[valid_idx, m_stdev] = sqrt(mvn_out$var)
  model[valid_idx, m_vs30] = exp(values_log) * exp(aak_resid)
  
  return(model)
}


weighting_run = function(model, stdev_weight=F, k=1) {
  model$vs30 = NA
  model$stdev = NA
  
  valid_idx = intersect(which(!is.na(model$aak_vs30)), which(!is.na(model$yca_vs30)))
  if (length(valid_idx) == 0) {return(model)}

  log_a = log(model[valid_idx, "aak_vs30"])
  log_y = log(model[valid_idx, "yca_vs30"])
  if (stdev_weight) {
    m_a = (model[valid_idx, "aak_stdev"] ^ 2) ^ -k
    m_y = (model[valid_idx, "yca_stdev"] ^ 2) ^ -k
    w_a = m_a / (m_a + m_y)
    w_y = m_y / (m_a + m_y)
  } else {
    w_a = 0.5
    w_y = 0.5
  }
  log_ay = log_a * w_a + log_y * w_y
  model[valid_idx, "vs30"] = exp(log_ay)

  sig1sq = model[valid_idx, "aak_stdev"] ^ 2
  sig2sq = model[valid_idx, "yca_stdev"] ^ 2
  # Reference: https://en.wikipedia.org/wiki/Mixture_distribution#Moments
  model[valid_idx, "stdev"] = (w_a * ((log_a - log_ay) ^ 2 + sig1sq) + w_y * ((log_y - log_ay) ^ 2 + sig2sq)) ^ 0.5

  return(model)
}


###
### CUSTOM POINTS VERSION
###

# pick a method to load stations
#xy = data.frame(x=c(174.780278, 177), y=c(-41.300278, -37.983333))
#source("validation_stations.R")
#xy = SpatialPoints(read.table("/nesi/project/nesi00213/StationInfo/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll"))

#coordinates(xy) = ~ x + y
#crs(xy) = WGS84
#model = data.frame(spTransform(xy, NZTM))

#model = geology_model_run(model)
#model = terrain_model_run(model)
#model = mvn_run(model, vspr_aak, variogram_aak, "aak")
#model = mvn_run(model, vspr_yca, variogram_yca, "yca")
#model = weighting_run(model)
#write.csv(model, paste0(OUT, "/", "custom_points.csv"))


###
### WHOLE NZ
###

# don't use this many cores in cluster, leave for other users/processes
leave_cores = 0
# which models to generate
geology = T
terrain = T
job_size = 3000

cat("loading points...\n")
# original grid
xy00 = sp::makegrid(as(raster::extent(1000050, 2126350, 4700050, 6338350), "SpatialPoints"), cellsize=100)
# small christchurch centred grid for testing
#xy00 = sp::makegrid(as(raster::extent(1420050, 1639550, 5064950, 5294550), "SpatialPoints"), cellsize=400)
colnames(xy00) = c("x", "y")
cluster_model = split(x=data.frame(xy00), f=ceiling(seq(1, dim(xy00)[1])/job_size))
rm(xy00)

if (! file.exists(OUT)) {dir.create(OUT)}

# each instance of cluster uses about input data size * 2 RAM


### STEP 1: GEOLOGY MODEL
if (geology) {
  cat("geology model loading resources into cluster...\n")
  pool = makeCluster(detectCores() - leave_cores)
  # coast dataset: ~7MB/core, slope dataset: ~110MB/core, ahdiak gid dataset ~290MB/core
  clusterExport(cl=pool, varlist=c("coast_distance", "coast_poly", "coast_line", "NZTM", "NZMG",
                                   "slp_nzni_9c", "slp_nzsi_9c", "aak_map"))
  cat("running geology model...\n")
  t0 = Sys.time()
  cluster_model = parLapply(cl=pool, X=cluster_model, fun=geology_model_run)
  t1 = Sys.time()
  stopCluster(pool)
  cat("Geology model complete.\n")
  print(t1 - t0)
}


### STEP 2: TERRAIN MODEL
if (terrain) {
  cat("terrain model loading resources into cluster...\n")
  pool = makeCluster(detectCores() - leave_cores)
  # iwahashipike dataset: ~700MB/core
  clusterExport(cl=pool, varlist=c("NZTM", "iwahashipike"))
  cat("running terrain model...\n")
  t0 = Sys.time()
  cluster_model = parLapply(cl=pool, X=cluster_model, fun=terrain_model_run)
  t1 = Sys.time()
  stopCluster(pool)
  cat("Terrain model complete.\n")
  print(t1 - t0)
}


### STEP 3: MVN
cat("starting mvn cluster...\n")
pool = makeCluster(detectCores() - leave_cores)
clusterExport(cl=pool, varlist=c("numVGpoints", "useNoisyMeasurements", "covReducPar",
                                 "useDiscreteVariogram", "useDiscreteVariogram_replace",
                                 "optimizeUsingMatrixPackage", "GEOLOGY", "TERRAIN"))
if (geology) {
  t0 = Sys.time()
  cluster_model = parLapply(cl=pool, X=cluster_model, fun=mvn_run, vspr_aak, variogram_aak, "aak")
  t1 = Sys.time()
  cat("Geology mvn complete.\n")
  print(t1 - t0)
}
if (terrain) {
  t0 = Sys.time()
  cluster_model = parLapply(cl=pool, X=cluster_model, fun=mvn_run, vspr_yca, variogram_yca, "yca")
  t1 = Sys.time()
  cat("Terrain mvn complete.\n")
  print(t1 - t0)
}
stopCluster(pool)


### STEP 4: WEIGHTED MVN
if (geology & terrain) {
  pool = makeCluster(detectCores() - leave_cores)
  cat("running geology and terrain combination...\n")
  t0 = Sys.time()
  cluster_model = parLapply(cl=pool, X=cluster_model, fun=weighting_run)
  t1 = Sys.time()
  cat("Geology and Terrain combination complete.\n")
  print(t1 - t0)
  stopCluster(pool)
}

### STEP 5: OUTPUT
# combine
cluster_model = do.call(rbind, cluster_model)
maps = colnames(cluster_model)[-which(colnames(cluster_model) %in% c("x", "y"))]
# write all columns into rasters / grids
for (z in maps) {
  grid = cluster_model[, c("x", "y", z)]
  names(grid) = c("x", "y", "z")
  coordinates(grid) = ~ x + y
  crs(grid) = NZTM
  grid = rasterFromXYZ(grid)
  plot(grid)
  writeRaster(grid, filename=paste0(OUT, "/", z, ".nc"), format="CDF", overwrite=TRUE)
}
rm(grid)

cat("Finished.\n")

# to convert topography files to nztm equiv
#t = raster(paste0(PLOTRES, "Topo/srtm_all_filt_nz.hdf5"))
#t = projectRaster(t, to=aak_vs30, method="ngb")
#writeRaster(t, filename="vs30map_i5.nc", format="CDF", overwrite=TRUE)

# plotting done by GMT script instead
#png("Rplot.png", height=12, width=9, res=600, units="in")
#   raster::plot(aak_vs30, maxpixels=(aak_vs30@ncols * aak_vs30@nrows))
#   mtext("Geology Model Vs30", line=0.5, cex=1)
#dev.off()
