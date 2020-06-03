#!/usr/bin/env Rscript

library(parallel) # cluster

source("shared.R")

###
### WHOLE NZ
###

# don't use this many cores in cluster, leave for other users/processes
leave_cores = 2
# which models to generate
geology = T
terrain = T
job_size = 3000
# outputs placed into this directory
OUT = "vs30out"

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
                                   "slp_nzni_9c", "slp_nzsi_9c", "aak_map", "GEOLOGY"))
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
  # uses slightly more ram than geology but much faster so could decrcease cores here if RAM issue
  pool = makeCluster(detectCores() - leave_cores)
  # iwahashipike dataset: ~700MB/core
  clusterExport(cl=pool, varlist=c("NZTM", "iwahashipike", "TERRAIN"))
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
  writeRaster(grid, filename=paste0(OUT, "/", z, ".nc"), format="CDF", overwrite=TRUE)
  writeRaster(grid, filename=paste0(OUT, "/", z, ".tiff"), format="GTiff", overwrite=TRUE)
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
