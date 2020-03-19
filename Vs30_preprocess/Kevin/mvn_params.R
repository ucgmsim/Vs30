# mvn_params.R

# defines parameters to be passed by lexical scoping
# ON 08 AUGUST IT TOOK APPROX 2 HOURS AND 20 MINUTES to perform MVN calcs on
# all of NZ. This was with resampleFlag = False, numVGpoints = 32 (piecewise-linear),
# maxPixels = 150.

# SET THE MODEL TO INTERPOLATE:



# MODEL  <- "AhdiYongWeighted1"
# 
# # which variogram to use?
# vgName <- "v3"

# Pick one of these - new model workflow - applying geostatistics BEFORE combining models.

MODEL  <- "AhdiAK_noQ3_hyb09c"
vgName <- "v6"

# MODEL  <- "YongCA_noQ3"
# vgName <- "v7"





# Choose the value of covariance reduction parameter, "a". See script cov_red_coeff.R
# for details. (Set to zero to bypass covariance reduction entirely.)
covReducPar <- 1.5


MAKE_ALL_STDEV_ONEHALF <- F # diagnostic test: override input standard deviation raster values with all 0.5!
DIVIDE_RESID_BY_TWO <- F  # diagnostic test: divide residual surface by two.

# If culling data to a minimum distance criterion, set it here:
# (Set to zero to disable!)
# minThreshold <- 6000 # metres 
minThreshold <- 0 # metres 

resampleFlag <- F
optimizeUsingMatrixPackage <- T
newRes    <- 3200   # metres. For subsampling rasters (for testing).
newRes    <- 1600   # metres. For subsampling rasters (for testing).
useDiscreteVariogram <- F   # trying discrete distance values to improve computational speed.
useDiscreteVariogram_replace <- F # ONLY valid for useDiscreteVariogram = TRUE. Replace distances with rounded values and perform function evaluation by table lookup rather than simple interpolation function.
numVGpoints  <- 128     # number of variogram evaluation points for piecewise-linear (if useDiscreteVariogram=False) or piecewise constant (useDiscreteVariogram=True).
maxPixels <- 150

leaveFreeCores <- 1 # leave this many cores free during parallel work

# whether or not to use noisy measurements as described in Wea Equations 25 to 42.
# Note that this requires lnMeasUncer to be included with observations (i.e. vspr dataframe)
useNoisyMeasurements <- T
# useNoisyMeasurements <- F  # 20180316 - diagnostic


# Timing notes
# newRes=1600 -- 110 seconds (on Hypocentre)




# Alternately, compute maxPixels based on estimate of required RAM.
# # maxPixels ------------------
# # maximum number of pixels to perform MVN with.
# # This is the main value that limits RAM usage!
# # Reasonable value calculation:
# RAM_GB_hypocentre <- 1 # RAM limit on 32-core machine Hypocentre
# RAM_GB_microfunk  <- 0.1 # RAM limit on laptop Microfunk
# RAMavailableGB <- RAM_GB_microfunk # Choose max GB of RAM to use
# numCores <- 2 # must change this manually
# RAMperCoreGB <- RAMavailableGB / numCores
# RAMperCoreB <- RAMperCoreGB * 1e9   # bytes
# 
# # Assume double precision values stored on NxN covariance 
# # matrix, so memory = 2 * (maxPixels^2) bytes:
# maxPixels <- floor(sqrt(RAMperCoreB / 2))
# maxPixels <- 2000


# save a list of parameters (so it can be fed
#to clusterExport() later when running parallel operations.)
paramsList <- ls()
