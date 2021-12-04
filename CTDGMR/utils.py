import ot
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.utils import check_random_state
from scipy.stats import norm

def log_normal(diffs, covs, prec=False):
    """
    log normal density of a matrix X
    evaluated for multiple multivariate normals
    =====
    input:
    diffs: array-like (N, M, d)
    covs: array-like (N, M, d, d)
    prec: if true, return the precision matrices
    """
    n, m, d, _ = covs.shape
    if d == 1:
        precisions_chol = (np.sqrt(1 / covs)).reshape((n * m, d, d))
    else:
        precisions_chol = np.empty((n * m, d, d))
        for k, cov in enumerate(covs.reshape((-1, d, d))):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                raise ValueError("covariance chol is wrong.")
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(d),
                                                         lower=True).T
    log_det = (np.sum(np.log(precisions_chol.reshape(n * m, -1)[:, ::d + 1]),
                      1))
    diffs = diffs.reshape((-1, d))
    y = np.einsum('ij,ijk->ik', diffs, precisions_chol)
    log_prob = np.sum(np.square(y), axis=1)
    log_probs = -.5 * (d * np.log(2 * np.pi) + log_prob) + log_det
    if prec:
        precisions = np.zeros_like(precisions_chol)
        for k, precision_chol in enumerate(precisions_chol):
            precisions[k] = precision_chol.dot(precision_chol.T)
        return log_probs.reshape((n, m)), precisions
    else:
        return log_probs.reshape((n, m))


def rmixGaussian(means, covs, weights, n_samples, random_state=0):
    """
    Sample from a Gaussian mixture

    Parameters
    ----------
    means : array-like, shape (n, d)
    covs :  array-like, shape (n, d, d)
    weights : array-like, shape (n,)

    Returns
    -------
    # n_sampels of samples from the Gaussian mixture
    """
    if n_samples < 1:
        raise ValueError(
            "Invalid value for 'n_samples': %d . The sampling requires at "
            "least one sample." % (n_components))

    rng = check_random_state(random_state)
    n_samples_comp = rng.multinomial(n_samples, weights)
    X = np.vstack([
        rng.multivariate_normal(mean, cov, int(sample))
        for (mean, cov, sample) in zip(means, covs, n_samples_comp)
    ])

    y = np.concatenate([
        np.full(sample, j, dtype=int)
        for j, sample in enumerate(n_samples_comp)
    ])

    return (X, y)


def df(x, mean, sd, f):
    x = x.reshape(-1, 1) - mean.T
    # x = x - mean.T
    x /= sd
    return f.pdf(x) / sd


def dmixf(x, mean, var, w, f):
    """
    Input: 
    x: array-like (n,)
    mean: array-like (k, )
    sd: array-like (k, )
    w: array-like (k, )
    Output:
    sum(w*pnorm(x,mean,sd)): array-like (n,)
    """
    sd = np.sqrt(var)
    prob = df(x, mean, sd, f)
    prob *= w
    return prob.sum(1)
