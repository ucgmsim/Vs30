
# for calculating distances with 2 vectors
library(proxy)

library(gstat)
library(Matrix)
library(raster)

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

    n_new = nrow(model_locations)
    minDist_m_log = log(0.1)
    maxDist_m_log = log(2000e3) # 2000 km
    logDelta = (maxDist_m_log - minDist_m_log) / (numVGpoints - 1)
    # distances to evaluate variogram (metres)
    distVec = exp(seq(minDist_m_log, maxDist_m_log, length.out=numVGpoints))
    # Correlation vs covariance can be summarized by examining Diggle & Ribeiro eq 5.6:
    # V_Y(u) = tau^2 + sigma^2 * {1 - rho(u)}.
    # It is clear from this form that rho(u) MUST be 1 at u=0 and MUST be continuously decreasing with u.
    # i.e., a discontinuity at u=0 is NOT going to yield the right answer here.
    # This makes sense intuitively since in this formulation it is measurement uncertainty, RATHER THAN a nonzero nugget,
    # that determines how "closely" the interpolation function should track the data.
    # covariance=T yields covariance function, gamma() in Diggle & Ribeiro ch3.
    correlationFunction = variogramLine(object=variogram, maxdist=maxDist_m, dist_vector=distVec, covariance=T)
    # normalize. for Matern style variograms, model$psill[2]
    correlationFunction$gamma = correlationFunction$gamma / variogram$psill[2]
    # is t the partial sill - aka "sigma^2" in Diggle & Ribeiro Equation 5.6.
    # if nonzero nugget, this removes discontinuity at origin. (not needed, distance vector starts at > 0)
    correlationFunction$gamma[correlationFunction$dist==0] = 1
    if (useDiscreteVariogram) interp_method = "constant" else interp_method = "linear"
    corrFn = approxfun(x=correlationFunction, rule=2, method=interp_method)
    # this will be < 1 because distance starts at > 0
    corr1 = correlationFunction$gamma[1]

    # Wea equation 33, 40, 41
    if (useNoisyMeasurements) {
        omegaObs = sqrt(modelVarObs / (modelVarObs + obs_stdev_log^2))
        obs_residuals = omegaObs * obs_residuals
    }
 
    # initialize outputs, pred and var
    pred = var = vector(mode="numeric", length=n_new)
    lpdf = data.frame(model_locations)
    for (i in seq(n_new)) {
        # point to observations
        distances = as.matrix(
            proxy::dist(as.matrix(lpdf[i,]), obs_locations, diag=TRUE, upper=TRUE)
        )
        # only observed which are within 10km to the point
        wanted = which(apply(distances, 2, FUN=min) < 10000)
        names(wanted) = NULL
        if (length(wanted) == 0) {
            # not close enough to any points
            pred[i] = modeledValues[i]
            var[i] = model_variances[i] * corr1
            next
        }

        # distances between all
        cov_matrix = as.matrix(stats::dist(x=rbind(lpdf[i,], obs_locations[wanted,]), diag=TRUE, upper=TRUE))
        # correlation function
        cov_matrix = array(sapply(cov_matrix, corrFn), dim=dim(cov_matrix))
        # uncertainties
        cov_matrix = cov_matrix * tcrossprod(sqrt(c(model_variances[i], modelVarObs[wanted])))

        if (useNoisyMeasurements) {
            omega = tcrossprod(c(1, omegaObs[wanted]))
            diag(omega) = 1
            cov_matrix = cov_matrix * omega
        }

        # covariance reduction factors
        if (covReducPar > 0) {
            cov_matrix = cov_matrix * exp(-covReducPar *
                    as.matrix(stats::dist(c(modeledValues[i], logModVs30obs[wanted]), 
                                diag=T, upper=T)))
        }

        cov_matrix[-1, -1] = solve(cov_matrix[-1, -1])
        pred[i] = modeledValues[i] +
            cov_matrix[1, -1] %*% cov_matrix[-1, -1] %*% obs_residuals[wanted]
        var[i] = cov_matrix[1, 1] -
            (cov_matrix[1, -1] %*% cov_matrix[-1, -1] %*% cov_matrix[-1, 1])
    }
    return(data.frame(pred, var))
}
