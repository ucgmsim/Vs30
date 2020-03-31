
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


geology_mvn_run = function(model, vspr_aak, variogram) {
  library(gstat)
  library(Matrix)
  library(raster)
  
  valid_idx = which(!is.na(model[, c("aak_values_log")]))
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
xya = array(c(176.8801,-39.6710,175.9910,-40.6713,176.9159,-39.4896,177.5278,-38.3340,
    176.8761,-39.4984,176.8411,-38.8523,175.8698,-40.3382,175.9611,-40.0586,176.8968,-39.5066,
    176.8720,-39.4687,177.4249,-39.0342,176.9149,-39.4859,173.8550,-39.4500,177.1109,-38.2592,
    173.7847,-41.2723,174.9544,-41.2058,174.8259,-41.1275,173.2574,-42.0880,174.9310,-41.2312,
    174.9211,-41.2335,173.2742,-41.2878,174.9538,-41.2023,174.1384,-41.8274,174.8218,-41.2320,
    174.8650,-41.2575,174.7784,-41.2799,174.7811,-41.2906,174.9193,-41.2323,174.7813,-41.2954,
    174.9401,-41.2074,174.8794,-41.2245,172.6119,-40.5570,173.3795,-41.2171,175.0651,-41.1264,
    174.9042,-41.2294,174.6981,-41.2259,174.0764,-41.6723,174.9043,-41.2521,174.7739,-41.2987,
    173.6821,-42.4258,174.7682,-41.2840,174.9485,-41.2574,174.8184,-41.3149,175.0409,-41.1268,
    174.7055,-41.2654,172.7037,-41.4290,174.9548,-41.1804,174.7793,-41.2743,174.7763,-41.2792,
    173.9051,-41.4395,174.8315,-41.1249,174.7421,-41.2848,174.8603,-41.2230,174.9260,-41.1914,
    175.0050,-40.9143,174.9022,-41.2470,173.2768,-41.2665,174.8376,-41.3264,174.8090,-41.3264,
    172.8305,-42.5232,172.5695,-43.5909,174.9855,-39.7546,172.6357,-43.6524,174.7746,-41.2722,
    173.2837,-41.2709,172.1165,-41.2494,172.7272,-43.6303,171.5998,-41.7557,172.9216,-42.9386,
    172.5433,-42.8695,172.8003,-42.7012,173.1285,-41.3892,172.6538,-43.7061,172.1803,-42.3346,
    172.9052,-41.7625,172.3813,-43.3123,172.3280,-41.7999,172.7052,-42.9631,172.6638,-43.3765,
    172.8026,-42.7594,173.0348,-42.9674,173.2749,-42.8135,171.8644,-42.1197,172.3363,-43.2622,
    172.6572,-43.5794,172.4680,-43.6232,171.5677,-42.9489,172.2710,-42.7817,172.7309,-43.1547,
    172.7314,-43.5069,176.9805,-38.6154,178.3066,-37.5623,178.2572,-38.0715,178.3008,-38.3728,
    178.3654,-37.6333,173.0095,-41.1247,174.9815,-41.1519,175.7908,-39.8078,174.8739,-41.2247,
    174.7037,-36.8223,177.2892,-38.0141,175.7956,-39.6800,175.4126,-39.4174,176.3111,-40.4009,
    176.5843,-39.9439,176.2210,-40.8988,175.5785,-40.6605,175.2483,-38.8641,178.0319,-38.6822,
    175.5931,-40.3629,175.6076,-40.3489,176.6118,-40.3022,169.1430,-44.6946,168.4065,-44.8644,
    170.0972,-43.7364,168.6609,-43.9962,169.7194,-46.2491,170.0983,-44.2546,169.2329,-44.2304,
    168.6629,-45.0322,171.9599,-43.5396,175.2340,-41.5891,171.9301,-43.3368,170.7368,-43.0744,
    172.6351,-43.5219,172.5291,-40.8255,171.8548,-43.3215,174.2742,-39.5851,174.5681,-39.2605,
    171.8635,-43.4577,172.8750,-41.6731,172.0258,-43.3912,170.7372,-43.0716,178.0177,-38.6418,
    170.5561,-43.1489,173.9814,-41.9557,172.2524,-43.8087,172.9635,-43.8109,172.0938,-43.8968,
    172.1979,-43.6675,172.0888,-43.5862,171.7236,-43.2265,171.2340,-43.9239,171.4728,-42.7267,
    172.9738,-43.6374,171.1356,-42.8917,172.6732,-43.5120,170.3268,-43.3160,171.4217,-43.8231,
    171.0539,-43.7146,171.7936,-43.7294,172.6069,-43.4928,172.6449,-43.6065,171.4023,-43.7046,
    170.3590,-43.2612,171.2043,-42.4578,171.4079,-42.5240,172.5300,-43.4832,171.6110,-43.8373,
    172.6474,-43.5381,172.0231,-43.7515,171.4441,-42.7245,172.3811,-43.5928,170.1842,-43.3891,
    170.8287,-44.0987,172.0383,-43.3259,175.2260,-40.4614,175.6337,-40.3303,174.0734,-39.0624,
    174.1905,-39.1562,175.0480,-39.9336,177.9216,-38.6257,172.7248,-43.6078,172.5928,-43.5575,
    172.6605,-43.4446,172.6199,-43.5293,172.6643,-43.5562,175.6478,-40.9504,175.7088,-40.6495,
    176.4670,-39.4334,174.7861,-41.2675,176.0675,-38.6863,175.8150,-38.9863,176.0937,-38.6325,
    167.4725,-46.1474,167.7651,-46.2868,168.2378,-45.6678,168.1184,-45.3665,167.9470,-45.8924,
    172.4954,-43.3694,170.0198,-43.4632,178.0227,-38.6665,174.7763,-41.2837,174.7742,-41.2816,
    175.4615,-41.2109,174.7848,-41.2931,175.1556,-38.3328,174.7585,-39.1243,175.5380,-38.3294,
    175.0930,-40.7894,171.0062,-44.3832,172.7115,-43.5585,176.9855,-37.9615,172.7693,-43.5784,
    172.6634,-43.5053,172.4720,-43.5500,172.5644,-43.5362,172.6828,-43.5258,171.9525,-41.8571,
              167.7191,-45.4167,#TAFS
              172.7507,-43.5679,#PARS
              172.6242,-43.5656,#CMHS
              171.6664,-43.5632,#MTHS
              172.6411,-43.5136,#STAS
              172.7568,-43.5692,#SUMS
              172.6162,-43.5067,#SACS
              172.6423,-43.4994,#MPSS
              172.6214,-43.5395,#MORS
              172.7256,-43.5847,#MTPS
              178.5482,-37.6888,#ECLS
              176.8429,-39.6458,#HCDS
              172.7180,-43.4954,#NNBS
              172.7297,-43.6086,#LCQC
              172.7808,-43.8224,#TOKS
              176.3438,-38.4380,#RPCS
              169.6311,-45.9540,#TUZ
              175.8378,-39.7464,#UTKS
              174.9266,-41.2050,#LNBS
              173.5390,-42.4160,#KHZ
              170.9691,-45.0997,#OAMS
              170.1844,-44.3856,#LBZ
              170.4706,-45.9052,#DUNS
              166.6859,-45.6876,#RLNS
              175.2300,-41.0769,#KIRS
              174.8936,-41.2308,#LHUS
              175.5018,-37.7310,#TOZ
              175.9097,-38.4689,#TIRS
              173.0536,-42.6195,#WTMC
              176.1567,-37.7027,#TBCS
              174.7845,-41.2781,#WSTS
              176.2552,-38.1357,#ROTS
              174.7260,-36.8853,#MAGS
              173.8757,-41.7149,#BSWZ
              174.2138,-41.7490,#CMWZ
              174.7705,-36.8532,#AKUS
              174.8719,-36.9756,#PTHS
              172.6139,-43.4675,#SMTC
              170.9640,-42.7169,#HMCS
              172.7148,-43.6084,#LPOC
              172.5110,-42.5942,#GLWS
              175.3274,-41.1187,#FTPS
              174.9441,-41.2421,#ARKS
              172.5959,-43.2744,#ASHS
              172.1022,-43.4897,#DFHS
              174.7788,-41.2879,#FKPS
              171.7476,-43.9024,#ADCS
              175.5730,-40.2135,#FAHS
              172.7094,-43.5798,#HVSC
              176.1014,-40.2017,#DVHS
              175.0359,-41.1399,#HIBS
              174.0231,-41.2796,#QCCS
              173.3509,-41.6202,#WVFS
              172.8211,-40.8494,#TSFS
              174.9159,-41.2043,#SOCS
              173.9444,-41.5077,#MGCS
              174.8461,-41.1385,#PFAS
              174.8391,-41.1314,#POLS
              174.8923,-41.1966,#LHBS
              175.1554,-41.0781,#TMDS
              174.9033,-41.2117,#LHES
              174.8932,-41.2047,#LHRS
              174.7831,-41.2754,#TFSS
              175.1438,-40.7549,#OTKS
              175.2794,-40.6216,#HOCS
              172.3175,-43.6751,#SLRC
              170.1957,-46.0718,#TMBS
              170.4929,-45.9022,#DKHS
              175.2951,-37.7856,#HBHS
              170.5097,-45.8986,#SKFS
              174.7755,-41.3008,#WNHS
              175.0854,-41.1049,#TOTS
              174.9002,-41.2894,#EBPS
              172.7022,-43.5016,#HPSC
              175.5465,-37.1397,#TMHS
              171.8046,-41.7450,#DSZ
              167.9264,-44.6733,#MSZ
              175.7526,-39.4069,#MWDS
              177.6737,-39.0218,#KNZ
              174.7744,-41.2955,#CUBS
              169.0441,-43.8830,#HDWS
              169.1388,-46.5369,#SYZ
              172.6822,-43.4804,#BWHS
              170.1930,-44.5630,#BENS
              170.6446,-45.0440,#ODZ
              170.4606,-44.0133,#TKAS
              168.9417,-46.1027,#GORS
              168.3784,-46.5868,#NZAS
              170.1469,-44.1910,#PKIS
              177.1504,-38.8070,#TUDS
              172.6406,-43.5269,#KILS
              172.5592,-43.5113,#MRSS
              176.7093,-38.0835,#KAFS
              169.4231,-43.7147,#LPLS
              176.2517,-39.7727,#WAKS
              176.6689,-37.8591,#MWFS
              167.1535,-45.4647,#DCZ
              175.8224,-40.6005,#EKS3
              175.9910,-40.6713,#EKS1
              175.8069,-40.7434,#EKS2
              174.9341,-41.2490,#PRKS
              177.7826,-37.7561,#HAZ
              167.2781,-45.5212,#MANS
              173.6293,-41.5397,#RCS1
              172.7327,-43.5579,#RCBS
              171.2104,-42.4492,#GMTS
              174.7812,-41.2965,#RQGS
              166.6807,-46.1663,#PYZ
              172.7569,-43.5714,#RHBS
              172.6247,-43.6325,#D14C
              175.6928,-37.5420,#TACS
              175.7698,-37.8159,#MMCS
              175.7455,-38.3521,#MTDS
              172.3554,-43.4214,#EYRS
              172.6275,-43.5359,#CHHC
              172.6326,-43.5323,#D09C
              169.3083,-45.2311,#EAZ
              170.5978,-45.8844,#OPZ
              169.0176,-44.8270,#WKZ
              169.8155,-43.5321,#FOZ
              168.7855,-44.0732,#JCZ
              176.8621,-40.0306,#PXZ
              175.7209,-36.7452,#KUZ
              176.2462,-40.6796,#BFZ
              174.2763,-41.2104,#TCW
              172.6232,-43.5747,#CRLZ
              173.9216,-40.8017,#DUWZ
              172.7256,-43.5847,#D15C
              171.2444,-44.4028,#TRCS
              171.9289,-43.3380,#SPFS
              176.4925,-39.1657,#BKZ
              168.3469,-46.4116,#ICCS
              172.6327,-43.5324,#D08C
              172.6142,-43.5402,#D06C
              174.3450,-35.9386,#WCZ
              175.4578,-36.2502,#GRZ
              176.2545,-38.1362,#ROPS
              170.3570,-44.6552,#AVIS
              173.5961,-35.2197,#OUZ
              172.6449,-43.6065,#D13C
              172.6327,-43.5324,#D10C
              172.6732,-43.5120,#D04C
              176.8322,-37.9777,#EDAS
              170.5139,-45.8626,#DGNS
              170.5022,-45.8743,#DCDS
              176.9189,-39.4884,#NAMS
              172.6757,-43.5173,#D07C
              172.1026,-43.6005,#COLD
              172.2013,-43.5958,#TLED
              172.2777,-43.5854,#KVSD
              174.7042,-41.3087,#SNZO
              174.6300,-39.7571,#WVHS
              177.0083,-38.1476,#RUAS
              167.4694,-44.7949,#BCOF
              170.0952,-43.7332,#MCPS
              174.9045,-41.2526,#PHFS
              167.9255,-44.6730,#MSZS
              176.7309,-37.9280,#TA3S
              177.4147,-39.0356,#WICS
              176.8620,-40.0297,#PWZ
              176.6689,-37.8591,#TA2S
              176.7575,-37.8888,#TA1S
              175.2279,-41.5745,#NMPS
              175.8014,-39.6768,#THPS
              166.8848,-45.2207,#SECF
              168.6118,-43.9723,#JCWJ
              166.9727,-45.2026,#DECF
              171.6244,-42.7682,#AICS
              166.8693,-45.2931,#CSBF
              170.7359,-43.0747,#WVZS
              172.6323,-43.5322,#CCPS
              174.9147,-41.2366,#GPSS
              174.0764,-41.6722,#RCS2
              174.3169,-35.7150,#WBHS
              174.7746,-41.2700#POTR
))
xy = data.frame(x=c(175.283333), y=c(-37.783333)) # Hamilton
xy = data.frame(x=c(174.74), y=c(-36.840556)) # Auckland
xy = data.frame(x=c(174.780278, 177), y=c(-41.300278, -37.983333))
xy = data.frame(x=xya[c(T, F)], y=xya[c(F, T)])
xy = SpatialPoints(read.table("/nesi/project/nesi00213/StationInfo/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll"))

coordinates(xy) = ~ x + y
crs(xy) = WGS84
ahdiyong = mvn_points(xy, vspr_aak, vspr_yca, variogram, new_weight=F)
print(ahdiyong)


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
