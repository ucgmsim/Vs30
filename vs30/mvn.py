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
covReducPar = 1.5
obs_stdev_log = np.array([0.1, 0.1, 0.1])
numVGpoints = 128
useNoisyMeasurements = True
phi = 993  # terrain
phi = 1407  # geology


n_new = len(model_locations)
minDist_m_log = math.log(0.1)
maxDist_m_log = math.log(2000e3)  # 2000 km
distVec = np.exp(np.linspace(minDist_m_log, maxDist_m_log, numVGpoints))

corr = 1 / np.e ** (distVec / phi)
corrFn = lambda x: np.interp(x, distVec, corr)

# Wea equation 33, 40, 41
if useNoisyMeasurements:
    omegaObs = np.sqrt(modelVarObs / (modelVarObs + obs_stdev_log ** 2))
    # don't use *= to prevent overwriting function input
    obs_residuals = omegaObs * obs_residuals

# initialize outputs, pred and var
pred = np.copy(modeledValues)
var = model_variances * corr[0]
for i in range(n_new):
    # point to observations
    # distances = np.linalg.norm(obs_locations - model_locations[i], axis=1)
    dist_diffs = (obs_locations - model_locations[i]).T
    distances = np.sqrt(np.einsum("ij,ij->j", dist_diffs, dist_diffs))
    # only observed which are within 10km to the point
    wanted = distances < 10000
    if max(wanted) == False:
        # not close enough to any points
        continue

    # distances between all
    locs = np.vstack((model_locations[i], obs_locations[wanted]))
    locs = locs[:, 0] + 1j * locs[:, 1]
    cov_matrix = abs(locs[:, np.newaxis] - locs)
    # correlation function
    cov_matrix = np.interp(cov_matrix, distVec, corr)
    # uncertainties
    uncer = np.sqrt(np.insert(modelVarObs[wanted], 0, model_variances[i]))
    cov_matrix = cov_matrix * uncer[:, np.newaxis] * uncer

    if useNoisyMeasurements:
        omega = np.insert(omegaObs[wanted], 0, 1)
        omega = omega[:, np.newaxis] * omega
        omega[np.eye(len(omega), dtype=np.bool)] = 1
        cov_matrix *= omega

    # covariance reduction factors
    if covReducPar > 0:
        model = np.insert(logModVs30obs[wanted], 0, modeledValues[i])
        cov_matrix *= np.exp(-covReducPar * np.abs(model[:, np.newaxis] - model))

    cov_matrix2 = np.linalg.solve(cov_matrix[1:, 1:], np.eye(len(cov_matrix) - 1))
    pred[i] = modeledValues[i] + np.dot(
        np.dot(cov_matrix[0, 1:], cov_matrix2), obs_residuals[wanted]
    )
    var[i] = cov_matrix[0, 0] - np.dot(
        np.dot(cov_matrix[0, 1:], cov_matrix2), cov_matrix[1:, 0]
    )

print(pred, var)
