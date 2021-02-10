import math

import numpy as np

obs_locations = np.array([
    [1997000, 5502000],
    [2000000, 5500000],
    [2200000, 5800000]])
model_locations = np.array([
    [1999000, 5500000],
    [2000000, 5500000],
    [2002000, 5500000],
    [2220000, 5500000]])
model_variances = np.array([0.2, 0.1, 0.5, 0.5]) ** 2
modeledValues = np.log(np.array([400, 500, 600, 800]))
modelVarObs = np.array([0.2, 0.1, 0.2]) ** 2
logModVs30obs = np.log(np.array([300, 500, 800]))
obs_residuals = np.log(np.array([50, 1500, 1500])) - logModVs30obs
covReducPar = 1.5
obs_stdev_log = np.array([0.1, 0.1, 0.1])
numVGpoints = 128
useDiscreteVariogram = False
useNoisyMeasurements = True
phi = 993 # terrain
phi = 1407 # geology


n_new = len(model_locations)
minDist_m_log = math.log(0.1)
maxDist_m_log = math.log(2000e3) # 2000 km
distVec = np.exp(np.linspace(minDist_m_log, maxDist_m_log, numVGpoints))

corr = 1 / np.e ** (distVec / phi)
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
    lpdf = data.frame(model_locations, row.names=NULL)
    for (i in seq(n_new)) {
        # point to observations
        distances = as.matrix(
            proxy::dist(as.matrix(lpdf[i,]), obs_locations, diag=TRUE, upper=TRUE)
        )
        # only observed which are within 10km to the point
        wanted = which(apply(distances, 2, FUN=min) < 10000)
        names(wanted) = NULL
        if len(wanted) == 0:
            # not close enough to any points
            pred[i] = modeledValues[i]
            var[i] = model_variances[i] * corr[0]
            continue

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
