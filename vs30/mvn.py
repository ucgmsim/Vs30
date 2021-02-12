import math

import numpy as np

obs_locations = np.array([[1997000, 5502000], [2000000, 5500000], [1998000, 5501000]])
model_locations = np.array(
    [[1999000, 5500000], [2000000, 5500000], [2002000, 5500000], [2220000, 5500000]]
)
model_variances = np.array([0.2, 0.1, 0.5, 0.5]) ** 2
modeledValues = np.log(np.array([400, 500, 600, 800]))
modelVarObs = np.array([0.2, 0.1, 0.2]) ** 2
logModVs30obs = np.log(np.array([300, 500, 800]))
obs_residuals = np.log(np.array([50, 1500, 1500])) - logModVs30obs
obs_stdev_log = np.array([0.1, 0.1, 0.1])

cov_reduc = 1.5
use_noisy_measurements = True
phi = 993  # terrain
phi = 1407  # geology
min_dist = 0.1 # 10 cm
max_dist = 10034.4365 # 10 km, keep results same by using one of the interp values 128points/2000e3 max_dist
dist_points = 88


def mvn(sites, phi, cov_reduc=1.5, use_noisy_measurements=True, min_dist=0.1, max_dist=10034.4365, dist_points=88):
    dist = np.exp(np.linspace(math.log(min_dist), math.log(max_dist), dist_points))
    corr = 1 / np.e ** (dist / phi)
    
    # Wea equation 33, 40, 41
    if use_noisy_measurements:
        omegaObs = np.sqrt(modelVarObs / (modelVarObs + obs_stdev_log ** 2))
        # don't use *= to prevent overwriting function input
        obs_residuals = omegaObs * obs_residuals
    
    # default outputs if no sites closeby
    pred = np.copy(modeledValues)
    var = model_variances * corr[0]
    
    for i in range(len(model_locations)):
        # model point to observations
        # distances = np.linalg.norm(obs_locations - model_locations[i], axis=1)
        dist_diffs = (obs_locations - model_locations[i]).T
        distances = np.sqrt(np.einsum("ij,ij->j", dist_diffs, dist_diffs))
        wanted = distances < max_dist
        if max(wanted) == False:
            # not close enough to any points
            continue
    
        # distances between all of interest
        locs = np.vstack((model_locations[i], obs_locations[wanted]))
        locs = locs[:, 0] + 1j * locs[:, 1]
        cov_matrix = abs(locs[:, np.newaxis] - locs)
        # correlation function - use interp to keep same behaviour at edges
        cov_matrix = np.interp(cov_matrix, dist, corr)
        # uncertainties
        uncer = np.sqrt(np.insert(modelVarObs[wanted], 0, model_variances[i]))
        cov_matrix = cov_matrix * uncer[:, np.newaxis] * uncer
    
        if use_noisy_measurements:
            omega = np.insert(omegaObs[wanted], 0, 1)
            omega = omega[:, np.newaxis] * omega
            omega[np.eye(len(omega), dtype=np.bool)] = 1
            cov_matrix *= omega
    
        # covariance reduction factors
        if cov_reduc > 0:
            model = np.insert(logModVs30obs[wanted], 0, modeledValues[i])
            cov_matrix *= np.exp(-cov_reduc * np.abs(model[:, np.newaxis] - model))
    
        inv_matrix = np.linalg.inv(cov_matrix[1:, 1:])
        pred[i] = modeledValues[i] + np.dot(
            np.dot(cov_matrix[0, 1:], inv_matrix), obs_residuals[wanted]
        )
        var[i] = cov_matrix[0, 0] - np.dot(
            np.dot(cov_matrix[0, 1:], inv_matrix), cov_matrix[1:, 0]
        )
    
    return(pred, var)
