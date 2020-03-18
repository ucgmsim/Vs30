
library(gstat)
library(Matrix)
library(raster)

source("R/mvn_params.R")

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

library(matrixcalc)

mvn = function(obs_locations, model_locations, model_variances, model,
               modeledValues, modelVarObs, obs_residuals,
               covReducPar, logModVs30obs, obs_stdev_log) {
    # obs_locations: locations of observations. (i.e., coordinates(vspr)
    # model_locations: locations where prediction & sigma estimates are desired
    # model_variances: gives model variance information to the MVN method
    # variogram: variogram table
    # modeledValues: interpolate on *model* values rather than *residuals.* The MVN is unaffected
    #                    (since sigma is linear) but it more closely resembles the formulation in the MVN paper.
    #                    In order to do this properly, modeledValues for prediction locations (i.e.: mu_Y1 in Worden
    #                    et al. parlance) must be provided as inputs.
    # modelVarObs: The *model* variance (not measurement uncertainty) corresponding to each of the observations.
    # residuals: The residuals (in log space) i.e. ln(obs/pred)
    # covReducPar: "Covariance reduction parameter," "a", is used to generate a "covariance 
    #              reduction factor." See script cov_red_coeff.R. Covariance reduction factor
    #              is between 0 and 1, and used to reduce the value of rho (correlation
    #              coefficient) to reduce the impact of MVN interpolation/extrapolation across
    #              geologic boundaries.
    #              0 for no covariance reduction.
    # logModVs30obs: log of modeled values of Vs30 for observed locations. (log of modeled Vs30
    #                for the prediction locations is passed in as "modeledValues".)
    #                Only used if covReducPar > 0.
    #
    # output: dataframe with columns "var1.pred" and "var1.var" (prediction and variance).

    # interpolation is done on the residuals - i.e. in natural log space - not on the model predictions
    # therefore must be careful that transformation of variables is handled properly.

    ###############################################################################
    # prepare the components of the MVN formulation from Worden et al.
    #
    # Because of intractable size of covariance matrix (whose size increases as the
    # square of the number of pixels being predicted), the prediction needs to be
    # repeated many times for a subset of the pixels of interest.
    # Below, two groups of variables are prepared: the ones whose values do not change
    # (e.g. everything associated with observations), and the ones whose values 
    # DO change for different pixels under consideration.
    # The latter group are implemented in a loop.

    n_obs = length(obs_residuals)
    n_new = nrow(model_locations)
    minDist_m_log = log(0.1)
    maxDist_m_log = log(2000e3) # 2000 km
    logDelta = (maxDist_m_log - minDist_m_log) / (numVGpoints - 1)
    # distances to evaluate variogram (metres)
    distVec = exp(seq(minDist_m_log, maxDist_m_log, length.out=numVGpoints))
    interpVec = c(-Inf, seq(minDist_m_log + 0.5*logDelta, maxDist_m_log + 0.5*logDelta, length.out=numVGpoints))
    # Correlation vs covariance can be summarized by examining Diggle & Ribeiro eq 5.6:
    # V_Y(u) = tau^2 + sigma^2 * {1 - rho(u)}.
    # It is clear from this form that rho(u) MUST be 1 at u=0 and MUST be continuously decreasing with u.
    # i.e., a discontinuity at u=0 is NOT going to yield the right answer here.
    # This makes sense intuitively since in this formulation it is measurement uncertainty, RATHER THAN a nonzero nugget,
    # that determines how "closely" the interpolation function should track the data.
    correlationFunction = variogramLine(object=variogram, maxdist=maxDist_m, dist_vector=distVec, covariance=T)  # covariance=T yields covariance function, gamma() in Diggle & Ribeiro ch3.
    # now normalize:
    correlationFunction$gamma = correlationFunction$gamma / variogram$psill[2] # for Matern style variograms, model$psill[2] 
    # is t the partial sill - aka "sigma^2" in 
    # Diggle & Ribeiro Equation 5.6.
    correlationFunction$gamma[correlationFunction$dist==0] = 1 # if nonzero nugget, this removes discontinuity at origin. (not really needed since distance vector starts at distance > 0.)
    # rule 2 for nearest
    if (useDiscreteVariogram) interp_method = "constant" else interp_method = "linear"
    corrFn = approxfun(x=correlationFunction, rule=2, method=interp_method)
  
    # only used if useDiscreteVariogram_replace=TRUE!
    # Create a lookup table containing discrete distances (m) and their covariance values:
    variogramTable = data.frame(distanceMetres=distVec, correlation=corrFn(distVec))
    
    
    if (useNoisyMeasurements) {
      # Wea equation 33, 40, 41
      omegaObs = sqrt(modelVarObs / (modelVarObs + obs_stdev_log^2))
      obs_resibuals = omegaObs * obs_residuals
    }
    
    
    #### here are the changing values #######################
    # 300 seems to still speed up over smaller chunk sizes
    maxPixels = 300
    # split location list into chunks of maximum size maxPixels
    lpdf = data.frame(model_locations)
    sequence = seq(1, n_new)
    locPredChunks = split(x=lpdf, f=ceiling(sequence/maxPixels))
    modelVarPredChunks = split(x=model_variances, f = ceiling(sequence/maxPixels))
    modeledValuesChunks = split(x=modeledValues, f = ceiling(sequence/maxPixels))
    # initialize outputs, pred and var
    pred = var = c()

    for (i in seq(ceiling(n_new/maxPixels))) {
        if (optimizeUsingMatrixPackage) {
            locPredChunk = Matrix(data = data.matrix(locPredChunks[[i]]))
        } else {
            locPredChunk = data.matrix(locPredChunks[[i]])
        }
        modeledValuesChunk = modeledValuesChunks[[i]]

        cov_matrix = makeCovMatrix(obs_locations, locPred=locPredChunk, modelVarObs,
                                   modelVarPred=modelVarPredChunks[[i]], corrFn,
                                   interpVec, distVec, variogramTable)
  
        if (useNoisyMeasurements) {
            J_Y_1 = rep(1, length(modelVarPredChunks[[i]])) # Wea equation 37
            omega = c(J_Y_1, omegaObs) # Wea equation 37
            Omega = omega %*% t(omega) # Wea equation 38
            diag(Omega) = 1 # Wea line 283

            cov_matrix = cov_matrix * Omega # Wea equation 39 (element-by-element product)
        }

        if (covReducPar > 0) {
            # Modify covariance matrix with covariance reduction factors
            logModVs30 = c(modeledValuesChunk, logModVs30obs) # vector of log(modeled Vs30) for all points
            lnVs30iVs30j = as.matrix(dist(logModVs30, diag=T, upper=T)) # this gives all pairwise abs(ln(obs)-ln(pred)) = abs(ln(obs/pred)).
            covReducMat = exp(-covReducPar*lnVs30iVs30j)
            cov_matrix = covReducMat * cov_matrix
        }

        # check for positive definiteness
        if (!is.positive.definite(as.matrix(cov_matrix))) {
            warning("Not positive definite.")
        }

        cov_Y2Y2_inverse = solve(Sigma_Y2Y2(covMatrix=cov_matrix, n_obs))
        pred = c(pred, as.numeric(mu_Y1_given_y2(modeledValuesChunk, covMatrix=cov_matrix, 
                                   cov_Y2Y2_inverse=cov_Y2Y2_inverse, residuals=obs_residuals)))
        var = c(var, diag(as.matrix(cov_Y1Y1_given_y2(cov_matrix, cov_Y2Y2_inverse))))
    }

    return(data.frame(pred, var))
}


makeCovMatrix = function(locObs, locPred, modelVarObs, modelVarPred,
                         corrFn, interpVec, distVec, variogramTable) {
  # Takes the following inputs:
  #   locObs, locations and observational data
  #   locPred, locations of desired prediction point(s)
  #   modelVarObs,  variance of predictive model at points of observation
  #   modelVarPred, variance of predictive model at prediction points
  #   corrFn, for i.e. rho_{Y_iY_j} - the spatial correlation of residuals - e.g. Equation 24 in Wea.
  # 
  # locations and variogram function need to have consistent units (i.e. metres)
  #
  #
  
  M = length(modelVarPred)
  N = length(modelVarObs)
  allPoints = rbind(locPred, locObs) # vector of all spatial points
  distMat1 = as.matrix(dist(x=allPoints, diag=TRUE, upper=TRUE))
  if (useDiscreteVariogram_replace) {
    lookupFun = function(logDist_m) {
      #  distVec[findInterval(x = logDist_m, vec = interpVec)]}
      findInterval(x=logDist_m, vec=interpVec)}

    # create rounded distance vector:
    distMat1 = array(sapply(X=as.matrix(log(distMat1)), FUN=lookupFun, simplify=T), dim=dim(distMat1))
  }
  if(optimizeUsingMatrixPackage) {
    distMat = Matrix(data = as.matrix(distMat1)) # distances among all points. (metres)
  } else {
    distMat = as.matrix(distMat1) # distances among all points. (metres)
  }
  
  if(!(useDiscreteVariogram_replace)) {
    # apply the correlation function for every pair of points
    # multiply by two because variogram = 1/2 * Var(u), see e.g. Diggle & Ribeiro equation 3.1
    # NB this does not seem to matter in outputs.
    rhoMat = array(sapply(distMat, corrFn), 
                    dim=dim(distMat)) # * 2 # i don't think multiplying by two is needed.
  } else {  # If using replacement method...
    rhoMat = array(data=0,dim=dim(distMat)) #initialise
    for (distN in seq(1, nrow(variogramTable))) {
      #thesePixels = distMat == variogramTable$distanceMetres[distN]
      thesePixels = distMat == distN
      rhoMat[which(thesePixels)] = variogramTable$correlation[distN]
    }
  }
  
  
  if(optimizeUsingMatrixPackage) {
    rhoMat = Matrix(data=rhoMat)
  }
  modelVariance = as.matrix(c(modelVarPred, modelVarObs)) # column vector of model uncertainty corresponding to allPoints vector.

  modelStdDev = sqrt(modelVariance)

  # NOTE rhoMat and modelStdDev represent the matrix rho_{Y_iY_j} and vector sigma_{Y} respectively - found in equation 7 in Wea.



  covMatrix = modelStdDev %*% t(modelStdDev) * rhoMat # Equation 7, Wea
  
  return(covMatrix)
}


Sigma_Y1Y1 = function(covMatrix, N) {
  # Returns Sigma_{Y_1Y_1} as defined in Wea Equation 3 - i.e. the partition of the covariance
  # matrix corresponding to the desired predictions.
  # 
  # Inputs:
  #   covMatrix = Sigma_Y, as computed by Wea Equation 7, implemented in function makeCovMatrix().
  #   N         = Number of observations.
  
  M_plus_N = nrow(covMatrix) # NB - It is assumed that covMatrix is SQUARE!  
  M = M_plus_N - N
  
  return(covMatrix[1:M, 1:M, drop=F])
}

Sigma_Y2Y2 = function(covMatrix, N) {
  # Returns Sigma_{Y_2Y_2} as defined in Wea Equation 3 - i.e. the partition of the covariance
  # matrix corresponding to the desired predictions.
  # 
  # Inputs:
  #   covMatrix = Sigma_Y, as computed by Wea Equation 7, implemented in function makeCovMatrix().
  #   N         = Number of observations.
  
  M_plus_N = nrow(covMatrix) # NB - It is assumed that covMatrix is SQUARE!  
  M = M_plus_N - N
  
  return(covMatrix[(M+1) : M_plus_N,
                   (M+1) : M_plus_N, drop = F])
}

Sigma_Y1Y2 = function(covMatrix, N) {
  # Returns Sigma_{Y_1Y_2} as defined in Wea Equation 3 - i.e. the partition of the covariance
  # matrix corresponding to the desired predictions.
  # 
  # Inputs:
  #   covMatrix = Sigma_Y, as computed by Wea Equation 7, implemented in function makeCovMatrix().
  #   N         = Number of observations.
  
  M_plus_N = nrow(covMatrix) # NB - It is assumed that covMatrix is SQUARE!  
  M = M_plus_N - N
  
  return(covMatrix[1:M,
                   (M+1) : M_plus_N, drop=F])
}

Sigma_Y2Y1 = function(covMatrix, N) {
  # Returns Sigma_{Y_2Y_1} as defined in Wea Equation 3 - i.e. the partition of the covariance
  # matrix corresponding to the desired predictions.
  # 
  # Inputs:
  #   covMatrix = Sigma_Y, as computed by Wea Equation 7, implemented in function makeCovMatrix().
  #   N         = Number of observations.
  
  M_plus_N = nrow(covMatrix) # NB - It is assumed that covMatrix is SQUARE!  
  M = M_plus_N - N
  
  return(covMatrix[(M+1) : M_plus_N,
                   1:M, drop=F])
}


mu_Y1_given_y2 = function(mu_Y1_unconditional, covMatrix, cov_Y2Y2_inverse, residuals) {
  # Implements equation 5 in Wea.
  # 
  # Takes inverse of observational covariance matrix as an input, 
  # because it should only be computed once.
  #
  N = length(residuals) # number of observations
  sY1Y2 = Sigma_Y1Y2(covMatrix=covMatrix, N=N)
  product = sY1Y2 %*% cov_Y2Y2_inverse %*% residuals
  mu_y1_giv_y2 = mu_Y1_unconditional + product
  return(mu_y1_giv_y2)
}

cov_Y1Y1_given_y2 = function(covMatrix_unconditional, cov_Y2Y2_inverse) {
  # Implements equation 6 in Wea.
  # 
  # Takes inverse of observational covariance matrix as an input, 
  # because it should only be computed once.
  #
  N = nrow(cov_Y2Y2_inverse) # number of observations
  SigmaY1Y1 = Sigma_Y1Y1(covMatrix=covMatrix_unconditional, N=N)
  SigmaY1Y2 = Sigma_Y1Y2(covMatrix=covMatrix_unconditional, N=N)
  SigmaY2Y1 = Sigma_Y2Y1(covMatrix=covMatrix_unconditional, N=N)
  product = SigmaY1Y2 %*% cov_Y2Y2_inverse %*% SigmaY2Y1
  covY1Y1givenY2 = SigmaY1Y1 - product

  return(covY1Y1givenY2)
}


mvnRaster = function(inpIndexRaster, inpModelRaster, inpStdDevRaster,
                      vspr, variogram, MODEL, covReducPar) {
    # Interpolates residuals and variance geographically for a raster input.
  
    # duplicate rasters for filling with outputs
    lnObsPred = std_dev = inpModelRaster
  
    # originally this function fails because of model_values_log not exisisting in all NA case
    if (all(as.vector(is.na(inpModelRaster)))) { return }
  
    # removes ocean points
    valid_idx = Which(!is.na(inpIndexRaster), cells=TRUE)
    # Some NA values can get into the MVN inputs and cause annoying problems. Be extra cautious.
    valid_idx = intersect(valid_idx, Which(!is.na(inpModelRaster), cells=TRUE))
    valid_idx = intersect(valid_idx, Which(!is.na(inpStdDevRaster), cells=TRUE))[552929]

    # run over these locations
    model_locations = coordinates(xyFromCell(inpModelRaster, valid_idx, spatial=TRUE))
    print(model_locations)
    model_values_log = log(inpModelRaster[valid_idx])
    model_variances = inpStdDevRaster[valid_idx]^2
  
    # observed locations model info
    obs_locations = coordinates(vspr)
    obs_values_log = log(vspr[[sprintf("Vs30_%s",MODEL)]])
    obs_variances = (vspr[[sprintf("stDv_%s",MODEL)]])^2
    obs_residuals = log(vspr$Vs30) - obs_values_log
    obs_stdev_log = vspr$lnMeasUncer

    mvn_out = mvn(obs_locations, model_locations, model_variances, variogram,
                    model_values_log, obs_variances, obs_residuals,
                    covReducPar, obs_values_log, obs_stdev_log)
    # pred_residuals??
    lnObsPred[valid_idx] = mvn_out$pred - model_values_log
    std_dev[valid_idx] = sqrt(mvn_out$var)
  
    print(sprintf("min, max (model_values_log): %s %s", min(model_values_log, na.rm=T), max(model_values_log, na.rm=T)))
    print(sprintf("min, max (mvn_out$pred): %s %s", min(mvn_out$pred, na.rm=T), max(mvn_out$pred, na.rm=T)))
    # what if all values are nan (above)??
    krigedHybOutput = (c(lnObsPred=lnObsPred, stdDev=std_dev))
    return(krigedHybOutput)
}


mvn_points = function(xy, vspr, variogram, MODEL) {
    # Interpolates residuals and variance geographically for a SpatialPoints input.
    # xy is SpatialPoints, eg:
    #xy = data.frame(x=, y=)
    #coordinates(xy) = ~ x + y
    #crs(xy) = "proj4 string"

    polys = over(xy, map_NZGD00)
    model_values_log = log(1)# convert model GIDs to vs30
    model_variances = 1^2# convert model GIDs to stdev

    # mask for available points
    # TODO: return at all locations, nans possible (no valid_index)
    valid_idx = Which(!is.na(polys$INDEX), cells=TRUE)
    valid_idx = intersect(valid_idx, Which(!is.na(vs30s), cells=TRUE))
    valid_idx = intersect(valid_idx, Which(!is.na(stdevs), cells=TRUE))

    # observed locations info
    obs_locations = coordinates(vspr)
    obs_values_log = log(vspr[[sprintf("Vs30_%s",MODEL)]])
    obs_variances = (vspr[[sprintf("stDv_%s",MODEL)]])^2
    obs_residuals = log(vspr$Vs30) - obs_values_log
    obs_stdev_log = vspr$lnMeasUncer

    mvn_out = mvn(obs_locations, xy, model_variances[valid_idx], variogram,
                  model_values_log[valid_idx], obs_variances, obs_residuals,
                  covReducPar, obs_values_log, obs_stdev_log)

    # what if all values are nan (above)??
    krigedHybOutput = (c(lnObsPred=(mvn_out$pred - model_values_log), stdDev=sqrt(mvn_out$var)))
    return(krigedHybOutput)
}


mvn_oneTile = function(x, y, vspr, variogram, MODEL) {
    tile_name = sprintf("x%02dy%02d", x, y)
    load(sprintf("Rdata/INDEX_NZGD00_allNZ_%s.Rdata", tile_name))
    exts = extent(indexRaster)
    print(exts)
    
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
    # index joined to prevent cropping
    # TODO: index is just the polygon index for all points, might be possible to generate on the fly
    vs30 = crop(vs30, exts)
    stdev = crop(stdev, exts)
    
    mvnOutput = mvnRaster(indexRaster, vs30, stdev, vspr, variogram, MODEL, covReducPar)
    mvnVs30 = vs30 * exp(mvnOutput$lnObsPred)
    print(mvnVs30[1])
  
    for (prefix in c("MVkrg", "MVres", "MVsdv")) {
        rast = switch(prefix,
                       MVkrg=mvnVs30,
                       MVres=mvnOutput$lnObsPred,
                       MVsdv=mvnOutput$stdDev)
        writeRaster(rast, filename=paste0("tmp/", prefix, "_NZGD00_allNZ_", tile_name, "_", MODEL,
                                          "_noisy", substr(as.character(useNoisyMeasurements), 1, 1),
                                          sprintf("_minDist%03.1fkm", minThreshold/1000),
                                          "_", vgName, sprintf("_crp%03.1f", covReducPar),
                                          ".tif"), format="GTiff", overwrite=TRUE)
    }
}


for (y in 0:15) {
    for (x in 0:10) {
        #print(paste(x, y))
    }
}
x = 2
y = 1
mvn_oneTile(x, y, vspr_noQ3, variogram, MODEL)