import math

import numpy as np


def corr_func(distances, phi):
    """
    Correlation function by distance.
    phi is 993 for terrain model, 1407 for geology
    """
    # originally linearly interpolated from logarithmically spaced distances:
    # d = np.exp(np.linspace(np.log(0.1), np.log(2000e3), 128))
    # c = 1 / np.e ** (d / phi)
    # return np.interp(distances, d, c)
    # minimum distance of 0.1 metres enforced
    return 1 / np.e ** (np.maximum(0.1, distances) / phi)


def tcrossprod(x):
    """
    Cross product from a 1d numpy array.
    """
    return x[:, np.newaxis] * x


def dists(x):
    """
    Euclidean distance from 2d diff array.
    """
    # return np.linalg.norm(x, axis=1)
    # alternative, may be faster
    return np.sqrt(np.einsum("ij,ij->i", x, x))


def xy2complex(x):
    """
    Convert array of 2D coordinates to array of 1D complex numbers.
    """
    c = x[:, 0].astype(np.complex64)
    c.imag += x[:, 1]
    return c


def dist_mat(x):
    """
    Distance matrix between coordinates (complex numbers) or simple values.
    """
    return np.abs(x[:, np.newaxis] - x)


def mvn(
    model_locs,
    model_vs30,
    model_stdv,
    sites,
    model,
    cov_reduc=1.5,
    noisy=True,
    max_dist=10000,
):
    """
    Modify model with observed locations.
    phi: correlation range factor, see corr_func()
    noisy: noisy measurements
    """
    if model == "geology":
        phi = 1407
    elif model == "terrain":
        phi = 993

    obs_locs = np.column_stack((sites.easting.values, sites.northing.values))
    obs_model_stdv = sites[f"{model}_uncertainty"].values
    obs_residuals = np.log(sites.vs30.values / sites[f"{model}_vs30"].values)

    # Wea equation 33, 40, 41
    if noisy:
        omega_obs = np.sqrt(
            obs_model_stdv ** 2 / (obs_model_stdv ** 2 + sites.uncertainty.values ** 2)
        )
        obs_residuals *= omega_obs

    # default outputs if no sites closeby
    pred = np.log(model_vs30)
    var = model_stdv ** 2 * corr_func(0, phi)

    # model point to observations
    for i in range(len(model_locs)):
        distances = dists(obs_locs - model_locs[i])
        wanted = distances < max_dist
        if max(wanted) == False:
            # not close enough to any observed locations
            continue

        # distances between interesting points
        cov_matrix = dist_mat(xy2complex(np.vstack((model_locs[i], obs_locs[wanted]))))
        # correlation
        cov_matrix = corr_func(cov_matrix, phi)
        # uncertainties
        cov_matrix *= tcrossprod(np.insert(obs_model_stdv[wanted], 0, model_stdv[i]))

        if noisy:
            omega = tcrossprod(np.insert(omega_obs[wanted], 0, 1))
            np.fill_diagonal(omega, 1)
            cov_matrix *= omega

        # covariance reduction factors
        if cov_reduc > 0:
            cov_matrix *= np.exp(
                -cov_reduc
                * dist_mat(
                    np.insert(
                        np.log(sites.loc[wanted, f"{model}_vs30"].values), 0, pred[i]
                    )
                )
            )

        inv_matrix = np.linalg.inv(cov_matrix[1:, 1:])
        pred[i] += np.dot(np.dot(cov_matrix[0, 1:], inv_matrix), obs_residuals[wanted])
        var[i] = cov_matrix[0, 0] - np.dot(
            np.dot(cov_matrix[0, 1:], inv_matrix), cov_matrix[1:, 0]
        )

    return model_vs30 * exp(pred - log(model_vs30)), np.sqrt(var)
