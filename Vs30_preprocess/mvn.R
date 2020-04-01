
library(gstat)
library(Matrix)
library(raster)
library(rgdal)
library(rgeos) # gDistance

source("Kevin/mvn_params.R")
source("Kevin/MODEL_AhdiAK_noQ3_hyb09c.R")
source("Kevin/MODEL_YongCA_noQ3.R")

MODEL_AAK = "AhdiAK_noQ3_hyb09c"
MODEL_YCA = "YongCA_noQ3"

WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
NZTM = "+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
NZMG = "+proj=nzmg +lat_0=-41 +lon_0=173 +x_0=2510000 +y_0=6023150 +units=m +no_defs +ellps=intl +towgs84=59.47,-5.04,187.44,0.47,-0.1,1.024,-4.5993"

# working files
load("~/big_noDB/geo/QMAP_Seamless_July13K_NZGD00.Rdata")
# save memory
gidmap00 = map_NZGD00[, (names(map_NZGD00) %in% c("groupID_AhdiAK"))]
rm(map_NZGD00, map_NZGD00.Cant)
load(file="~/VsMap/Rdata/nzsi_9c_slp.Rdata")
load(file="~/VsMap/Rdata/nzni_9c_slp.Rdata")
rm(slp_nzni_9c, slp_nzsi_9c)
IP = as(raster("~/big_noDB/topo/terrainCats/IwahashiPike_NZ_100m_16.tif"), "SpatialGridDataFrame")

coast_poly = readOGR(dsn = "/nesi/project/nesi00213/PlottingData/Paths/lds-nz-coastlines-and-islands/EPSG_2193", layer="nz-coastlines-and-islands-polygons-topo-1500k")
coast_line = as(coast_poly, "SpatialLinesDataFrame")
coast_distance = function(xy, km=T) {
  # xy: SpatialPoints with CRS epsg:2193 NZGD2000
  nxy = length(xy)
  result = rep(0.0, nxy)
  mask = which(!is.na(over(xy, coast_poly)$name))
  result[mask] = apply(gDistance(xy[mask,], coast_line, byid=T), 2, min)
  if (km) {return(result/1000.0)}
  return(result)
}

# vs site properties
load("~/VsMap/Rdata/vspr.Rdata")
vspr_noQ3 = vspr[(vspr$QualityFlag != "Q3" | nchar(vspr$StationID) == 3 | is.na(vspr$QualityFlag)),]

# remove points where MODEL predictions don't exist
# vs30 should really have been updated first
aak_na = which(is.na(vspr_noQ3[[paste0("Vs30_", MODEL_AAK)]]))
if(length(aak_na) > 0) {
  warning("some locations don't have predictions for this model")
  vspr_aak = vspr_noQ3[-aak_na,]
} else {
  vspr_aak = vspr_noQ3
}
yca_na = which(is.na(vspr_noQ3[[paste0("Vs30_", MODEL_YCA)]]))
if(length(yca_na) > 0) {
  warning("some locations don't have predictions for this model")
  vspr_yca = vspr_noQ3[-yca_na,]
} else {
    vspr_yca = vspr_noQ3
}
# update model values with geology model tweaks
vspr_aak_points = SpatialPoints(vspr_aak@coords, proj4string=vspr_aak@proj4string)
distances = coast_distance(vspr_aak_points)
xy49 = spTransform(vspr_aak_points, NZMG)
slp09c = xy49 %over% slp_nzsi_9c.sgdf
slp09c[is.na(slp09c)] = (xy49 %over% slp_nzni_9c.sgdf)[is.na(slp09c)]
slp09c[is.na(slp09c)] = 0.0
model_params = data.frame(vspr_aak@data$groupID_AhdiAK, slp09c, distances)
names(model_params) = c("groupID_AhdiAK", "slp09c", "coastkm")
vspr_aak[["Vs30_AhdiAK_noQ3_hyb09c"]] = AhdiAK_noQ3_hyb09c_set_Vs30(model_params, g06mod=T, g13mod=T)

# import variogram
load(sprintf("~/VsMap/Rdata/variogram_%s_%s.Rdata", MODEL, vgName))

library(matrixcalc)

mvn = function(obs_locations, model_locations, model_variances, variogram,
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

        #if (!is.positive.definite(as.matrix(cov_matrix))) {
        #    warning("Not positive definite.")
        #    # warning above doesn't seem to matter, possible solution below
        #    #cov_matrix = nearPD(cov_matrix)$mat
        #}

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
                   (M+1) : M_plus_N, drop=F])
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

mvn_points = function(xy, vspr_aak, vspr_yca, variogram, new_weight=F, k=1, geology=T, terrain=T) {
    # Interpolates residuals and variance geographically for a SpatialPoints input.
    # xy is SpatialPoints
    # new_weight: T to use new weighting model, set k to 2 or 1
  
    n_pts = length(xy)
    blank = rep(NA, n_pts)
    result = data.frame(vs30=blank, sigma=blank)
    
    # mvn uses EPSG:2193 (NZGD2000/NZTM)
    xy00 = spTransform(xy, NZTM)

    # model input - coast distance
    coastkm = coast_distance(xy00)
    # model input - slope
    xy49 = spTransform(xy, NZMG)
    slp09c = xy49 %over% slp_nzsi_9c.sgdf
    slp09c[is.na(slp09c)] = (xy49 %over% slp_nzni_9c.sgdf)[is.na(slp09c)]
    slp09c[is.na(slp09c)] = 0.0
    rm(xy49)

    # model input - group IDs, mask for unusable points
    if (geology) {
        gid = over(xy00, gidmap00)$groupID_AhdiAK
        # make sure datatype is correct
        if (length(xy) > 1) {
            gid_aak = as.character(gid)
        } else {
            gid_aak=lapply(gid, as.character)
        }
        valid_idx = intersect(which(!is.na(gid_aak)), which(gid_aak != "00_WATER"))
        rm(gid)
    } else {gid_aak=NA}
    if (terrain) {
        groupID_YongCA = xy00 %over% IP
        colnames(groupID_YongCA) = "ID"
        gid_yca = plyr::join(groupID_YongCA, IPlevels[[1]])$category
        terrain_idx = which(!is.na(gid_yca))
        if (geology) {
            valid_idx = intersect(valid_idx, terrain_idx)
        } else {
            valid_idx = terrain_idx
        }
        rm(terrain_idx, groupID_YongCA)
    } else {gid_yca=NA}

    model_params = data.frame(gid_aak, gid_yca, slp09c, coastkm)
    names(model_params) = c("groupID_AhdiAK", "groupID_YongCA_noQ3", "slp09c", "coastkm")
    model_params = model_params[valid_idx,]
    # run models
    if (geology) {
        aak_values_log = log(AhdiAK_noQ3_hyb09c_set_Vs30(model_params, g06mod=T, g13mod=T))
        aak_variances = AhdiAK_noQ3_hyb09c_set_stDv(model_params)^2
    }
    if (terrain) {
        yca_values_log = log(YongCA_noQ3_set_Vs30(model_params))
        yca_variances = YongCA_noQ3_set_stDv(model_params)^2
    }
    rm(gid_aak, gid_yca, slp09c, coastkm)

    coords = coordinates(xy00)[valid_idx,]
    rownames(coords) = NULL
    rm(xy00)

    if (geology) {
        aak_obs_locations = coordinates(vspr_aak)
        aak_obs_values_log = log(vspr_aak[["Vs30_AhdiAK_noQ3_hyb09c"]])
        aak_obs_variances = (vspr_aak[["stDv_AhdiAK_noQ3_hyb09c"]])^2
        aak_obs_residuals = log(vspr_aak$Vs30) - aak_obs_values_log
        aak_obs_stdev_log = vspr_aak$lnMeasUncer
        mvn_aak = mvn(aak_obs_locations, coords, aak_variances, variogram,
                      aak_values_log, aak_obs_variances, aak_obs_residuals,
                      covReducPar, aak_obs_values_log, aak_obs_stdev_log)
        aak_resid = mvn_aak$pred - aak_values_log
        aak_stdev = sqrt(mvn_aak$var)
        aak_vs30 = exp(aak_values_log) * exp(aak_resid)
        if (!terrain) {
            result$sigma[valid_idx] = aak_stdev
            result$vs30[valid_idx] = aak_vs30
            return(result)
        }
    }
    if (terrain) {
        yca_obs_locations = coordinates(vspr_yca)
        yca_obs_values_log = log(vspr_yca[["Vs30_YongCA_noQ3"]])
        yca_obs_variances = (vspr_yca[["stDv_YongCA_noQ3"]])^2
        yca_obs_residuals = log(vspr_yca$Vs30) - yca_obs_values_log
        yca_obs_stdev_log = vspr_yca$lnMeasUncer
        mvn_yca = mvn(yca_obs_locations, coords, yca_variances, variogram,
                      yca_values_log, yca_obs_variances, yca_obs_residuals,
                      covReducPar, yca_obs_values_log, yca_obs_stdev_log)
        yca_resid = mvn_yca$pred - yca_values_log
        yca_stdev = sqrt(mvn_yca$var)
        yca_vs30 = exp(yca_values_log) * exp(yca_resid)
        if (!terrain) {
            result$sigma[valid_idx] = yca_stdev
            result$vs30[valid_idx] = yca_vs30
            return(result)
        }
    }

    # weighted result
    log_a = log(aak_vs30)
    log_y = log(yca_vs30)
    if (new_weight) {
        m_a = (aak_stdev ^ 2) ^ -k
        m_y = (yca_stdev ^ 2) ^ -k
        w_a = m_a / (m_a + m_y)
        w_y = m_y / (m_a + m_y)
    } else {
        w_a = 0.5
        w_y = 0.5
    }
    log_ay = log_a * w_a + log_y * w_y
    result$vs30[valid_idx] = exp(log_ay)
    sig1sq = aak_stdev ^ 2
    sig2sq = yca_stdev ^ 2

    # Reference: https://en.wikipedia.org/wiki/Mixture_distribution#Moments
    result$sigma[valid_idx] = (w_a * ((log_a - log_ay) ^ 2 + sig1sq) + w_y * ((log_y - log_ay) ^ 2 + sig2sq)) ^ 0.5
    return(result)
}


geology_model_run = function(xy00) {
  # xy00: data.frame of NZTM points
  library(raster)
  library(rgeos) # gDistance
  
  source("Kevin/MODEL_AhdiAK_noQ3_hyb09c.R")
  
  xy00 = SpatialPoints(xy00)
  crs(xy00) = NZTM
  result = data.frame(xy00, NA, NA)
  colnames(result) = c("x", "y", "aak_values_log", "aak_variances")
  
  # large amount of memory for polygon dataset
  gid_aak = over(xy00, gidmap00)$groupID_AhdiAK
  # make sure datatype is correct
  # TODO: short integer based model to save memory (use int indexes instead of string)?
  if (length(xy00) > 1) {
    gid_aak = as.character(gid_aak)
  } else {
    gid_aak=lapply(gid_aak, as.character)
  }
  valid_idx = intersect(which(!is.na(gid_aak)), which(gid_aak != "00_WATER"))
  if (length(valid_idx) == 0) {return(result)}
  xy00 = xy00[valid_idx]
  gid_aak = gid_aak[valid_idx]
  
  # coastline distance (not much memory)
  coastkm = coast_distance(xy00)
  
  # slope (more memory required for rasters)
  xy49 = spTransform(xy00, NZMG)
  slp09c = xy49 %over% slp_nzsi_9c.sgdf
  slp09c[is.na(slp09c)] = (xy49 %over% slp_nzni_9c.sgdf)[is.na(slp09c)]
  slp09c[is.na(slp09c)] = 0.0
  rm(xy00, xy49)
  
  model_params = data.frame(gid_aak, slp09c, coastkm)
  rm(gid_aak, slp09c, coastkm)
  names(model_params) = c("groupID_AhdiAK", "slp09c", "coastkm")
  
  # run model
  result$aak_values_log[valid_idx] = log(AhdiAK_noQ3_hyb09c_set_Vs30(model_params, g06mod=T, g13mod=T))
  result$aak_variances[valid_idx] = AhdiAK_noQ3_hyb09c_set_stDv(model_params)^2
  
  return(result)
}


terrain_model_run = function(model) {

  source("Kevin/MODEL_YongCA_noQ3.R")

  xy00 = coordinates(model[, c("x", "y")])
  crs(xy00) = NZTM

  model$yca_values_log = NA
  model$yca_variances = NA
  
  gid_yca = over(xy00, IP)
  valid_idx = which(!is.na(gid_yca))
  if (length(valid_idx) == 0) {return(model)}
  # TODO: use ints in model, not strings
  colnames(gid_yca) = "ID"
  gid_yca = plyr::join(gid_yca, IPlevels[[1]])$category

  model_params = data.frame(gid_yca[valid_idx,])
  rm(gid_yca)
  names(model_params) = c("groupID_YongCA_noQ3")

  # run model
  model$yca_values_log[valid_idx] = log(YongCA_noQ3_set_Vs30(model_params))
  model$yca_variances[valid_idx] = YongCA_noQ3_set_stDv(model_params)^2
  
  return(model)
}


geology_mvn_run = function(model, vspr_aak, variogram) {
  library(gstat)
  library(Matrix)
  library(raster)
  
  valid_idx = which(!is.na(model[, c("aak_values_log")]))
  if (length(valid_idx) == 0) {
    names(model)[names(model) == "aak_values_log"] = "aak_vs30"
    names(model)[names(model) == "aak_variances"] = "aak_stdev"
    return(model)
  }
  coords = coordinates(model[valid_idx, c("x", "y")])
  rownames(coords) = NULL
  
  aak_obs_locations = coordinates(vspr_aak)
  aak_obs_values_log = log(vspr_aak[["Vs30_AhdiAK_noQ3_hyb09c"]])
  aak_obs_variances = (vspr_aak[["stDv_AhdiAK_noQ3_hyb09c"]])^2
  aak_obs_residuals = log(vspr_aak$Vs30) - aak_obs_values_log
  aak_obs_stdev_log = vspr_aak$lnMeasUncer
  aak_values_log = model[valid_idx, "aak_values_log"]
  mvn_aak = mvn(aak_obs_locations, coords, model[valid_idx, "aak_variances"], variogram,
                aak_values_log, aak_obs_variances, aak_obs_residuals,
                covReducPar, aak_obs_values_log, aak_obs_stdev_log)
  # save memory, overwrite instead of add columns
  names(model)[names(model) == "aak_values_log"] = "aak_vs30"
  names(model)[names(model) == "aak_variances"] = "aak_stdev"
  aak_resid = mvn_aak$pred - aak_values_log
  model$aak_stdev[valid_idx] = sqrt(mvn_aak$var)
  model$aak_vs30[valid_idx] = exp(aak_values_log) * exp(aak_resid)
  
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
#xy00 = spTransform(xy, NZTM)

#model = geology_model_run(xy00)
#model = geology_mvn_run(model, vspr_aak, variogram)
#ahdiyong = mvn_points(xy, vspr_aak, vspr_yca, variogram, new_weight=F)
#print(ahdiyong)


###
### WHOLE NZ
###

library(parallel)

# don't use this many cores in cluster, leave for other users/processes
leave_cores = 0
# which models to generate
geology = T
terrain = F
job_size = 3000

print("loading points...")
# original grid
xy00 = sp::makegrid(as(raster::extent(1000050, 2126350, 4700050, 6338350), "SpatialPoints"), cellsize=100)
# small christchurch centred grid for testing
#xy00 = sp::makegrid(as(raster::extent(1420050, 1639550, 5064950, 5294550), "SpatialPoints"), cellsize=400)
colnames(xy00) = c("x", "y")
location_chunks = split(x=data.frame(xy00), f=ceiling(seq(1, dim(xy00)[1])/job_size))
rm(xy00)


# each instance of cluster uses about input data size * 2


### STEP 1: GEOLOGY MODEL
if (geology) {
  print("running geology model...")
  pool = makeCluster(detectCores() - leave_cores)
  # coast dataset: ~7MB/core, slope dataset: ~110MB/core, ahdiak gid dataset ~290MB/core
  clusterExport(cl=pool, varlist=c("coast_distance", "coast_poly", "coast_line", "NZTM", "NZMG",
                                   "slp_nzni_9c.sgdf", "slp_nzsi_9c.sgdf", "gidmap00"))
  t0 = Sys.time()
  cluster_model = parLapply(cl=pool, X=location_chunks, fun=geology_model_run)
  t1 = Sys.time()
  stopCluster(pool)
  print("Geology model complete.")
  print(t1 - t0)
}


### STEP 2: TERRAIN MODEL
if (terrain) {
  print("running terrain model...")
}


### STEP 3: GEOLOGY MVN
if (geology) {
  print("running geology mvn...")
  pool = makeCluster(detectCores() - leave_cores)
  clusterExport(cl=pool, varlist=c("mvn", "numVGpoints", "useNoisyMeasurements", "covReducPar",
                                   "useDiscreteVariogram", "useDiscreteVariogram_replace",
                                   "optimizeUsingMatrixPackage", "makeCovMatrix", "Sigma_Y2Y2",
                                   "mu_Y1_given_y2", "Sigma_Y1Y2", "cov_Y1Y1_given_y2", "Sigma_Y1Y1",
                                   "Sigma_Y2Y1"))
  t0 = Sys.time()
  cluster_model = parLapply(cl=pool, X=cluster_model, fun=geology_mvn_run, vspr_aak, variogram)
  t1 = Sys.time()
  stopCluster(pool)
  print("Geology mvn complete.")
  print(t1 - t0)
}


### STEP 4: TERRAIN MVN
if (terrain) {
  print("running geology mvn...")
}


### STEP 5: WEIGHTED MVN
if (geology & terrain) {
  print("running geology and terrain combination...")
}

### STEP 6: OUTPUT
# combine
cluster_model = do.call(rbind, cluster_model)
# 
aak_vs30 = cluster_model[, c("x", "y", "aak_vs30")]
names(aak_vs30) = c("x", "y", "z")
coordinates(aak_vs30) = ~ x + y
crs(aak_vs30) = NZTM
aak_vs30 = rasterFromXYZ(aak_vs30)
writeRaster(aak_vs30, filename="geology_model.nc", format="CDF", overwrite=TRUE)
rm(aak_vs30)


# to convert topography files to nztm equiv
#t = raster("/nesi/project/nesi00213/PlottingData/Topo/srtm_all_filt_nz.hdf5")
#u = projectRaster(t, crs=NZTM, method="ngb")
#u = projectRaster(t, to=aak_vs30, method="ngb")

# plotting done by GMT script instead
#png("Rplot.png", height=12, width=9, res=600, units="in")
#   raster::plot(aak_vs30, maxpixels=(aak_vs30@ncols * aak_vs30@nrows))
#   mtext("Geology Model Vs30", line=0.5, cex=1)
#dev.off()
