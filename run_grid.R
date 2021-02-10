#!/usr/bin/env Rscript

library(parallel) # cluster

source("R/main.R")

###
### WHOLE NZ
###

# don't use this many cores in cluster, leave for other users/processes
leave_cores = 0
# which models to generate
geology = F
terrain = T
job_size = 3000
# outputs placed into this directory
OUT = "vs30out"

ncores = detectCores() - leave_cores

cat("loading points...\n")
# original grid
xy00 = sp::makegrid(as(raster::extent(1000050, 2126350, 4700050, 6338350), "SpatialPoints"), cellsize=100)
# small christchurch centred grid for testing
#xy00 = sp::makegrid(as(raster::extent(1420050, 1639550, 5064950, 5294550), "SpatialPoints"), cellsize=400)
colnames(xy00) = c("x", "y")
cluster_model = split(x=data.frame(xy00), f=ceiling(seq(1, dim(xy00)[1])/job_size))
rm(xy00); gc()

if (! file.exists(OUT)) {dir.create(OUT)}

# geology and terrain both have large data files
# remove when not necessary to save on RAM
clean_geology = function() {
    rm(coast_distance, coast_poly, coast_line, slp_nzni_9c, slp_nzsi_9c,
       aak_map, model_ahdiak_get_gid, envir=globalenv()); gc()
}
clean_terrain = function() {
    rm(yca_map, model_yongca_get_gid, envir=globalenv()); gc()
}

### STEP 1: GEOLOGY MODEL
if (geology) {
    clean_terrain()
    cat("geology model loading resources into cluster...\n")
    pool = makeCluster(ncores)
    # coast dataset: ~7MB/core, slope dataset: ~110MB/core, ahdiak gid dataset ~290MB/core
    clusterExport(cl=pool, varlist=c("coast_distance", "coast_poly", "coast_line", "NZTM", "NZMG",
                                     "slp_nzni_9c", "slp_nzsi_9c", "GEOLOGY",
                                     "aak_map", "model_ahdiak_get_gid", "model_ahdiak"))
    clean_geology()
    cat("running geology model...\n")
    t0 = Sys.time()
    # chunk.size 1 should take a couple of seconds so give it 4
    # default is large and unnecessarily uses too much RAM
    # use LoadBalanced version which doesn't keep every iterations RAM in use
    cluster_model = parLapplyLB(cl=pool, X=cluster_model, fun=geology_model_run, chunk.size=4)
    t1 = Sys.time()
    stopCluster(pool); gc()
    cat("Geology model complete.\n")
    print(t1 - t0)
} else {
    clean_geology()
}


### STEP 2: TERRAIN MODEL
if (terrain) {
    if (geology) source("R/model_yongca.R")
    cat("terrain model loading resources into cluster...\n")
    # uses slightly more ram than geology but much faster so could decrcease cores here if RAM issue
    pool = makeCluster(ncores)
    # iwahashipike dataset: ~700MB/core
    clusterExport(cl=pool, varlist=c("NZTM", "TERRAIN",
                                     "yca_map", "model_yongca_get_gid", "model_yongca"))
    clean_terrain()
    cat("running terrain model...\n")
    t0 = Sys.time()
    # terrain model is very simple, run large chunks to fill CPU capacity
    cluster_model = parLapplyLB(cl=pool, X=cluster_model, fun=terrain_model_run, chunk.size=100)
    t1 = Sys.time()
    stopCluster(pool); gc()
    cat("Terrain model complete.\n")
    print(t1 - t0)
}


### STEP 3: MVN
cat("starting mvn cluster...\n")
pool = makeCluster(ncores)
clusterExport(cl=pool, varlist=c("numVGpoints", "useNoisyMeasurements", "covReducPar",
                                 "useDiscreteVariogram", "GEOLOGY", "TERRAIN"))
if (geology) {
  t0 = Sys.time()
  cluster_model = parLapplyLB(cl=pool, X=cluster_model, fun=mvn_run, vspr_aak, variogram_aak, "aak")
  t1 = Sys.time()
  cat("Geology mvn complete.\n")
  print(t1 - t0)
}
if (terrain) {
  t0 = Sys.time()
  cluster_model = parLapplyLB(cl=pool, X=cluster_model, fun=mvn_run, vspr_yca, variogram_yca, "yca")
  t1 = Sys.time()
  cat("Terrain mvn complete.\n")
  print(t1 - t0)
}
stopCluster(pool); gc()


### STEP 4: WEIGHTED MVN
if (geology & terrain) {
  pool = makeCluster(ncores)
  cat("running geology and terrain combination...\n")
  t0 = Sys.time()
  cluster_model = parLapplyLB(cl=pool, X=cluster_model, fun=weighting_run)
  t1 = Sys.time()
  cat("Geology and Terrain combination complete.\n")
  print(t1 - t0)
  stopCluster(pool); gc()
}

### STEP 5: OUTPUT
# combine
cluster_model = do.call(rbind, cluster_model)
# this is required because R is a meme
gc()
maps = colnames(cluster_model)[-which(colnames(cluster_model) %in% c("x", "y"))]
# write all columns into rasters / grids
for (z in maps) {
  grid = cluster_model[, c("x", "y", z)]
  names(grid) = c("x", "y", "z")
  coordinates(grid) = ~ x + y
  crs(grid) = NZTM
  grid = rasterFromXYZ(grid)
  writeRaster(grid, filename=paste0(OUT, "/", z, ".nc"), format="CDF", overwrite=TRUE)
  writeRaster(grid, filename=paste0(OUT, "/", z, ".tif"), format="GTiff", overwrite=TRUE)
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
