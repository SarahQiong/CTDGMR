import ot
import time
import numpy as np
from scipy import linalg
from .distance import GMM_L2, GMM_CTD


def moment_preserving_merge(w1, mu1, cov1, w2, mu2, cov2):
    w11, w21 = w1 / (w1 + w2), w2 / (w1 + w2)
    mu = w11 * mu1 + w21 * mu2
    cov = w11 * cov1 + w21 * cov2 + w11 * w21 * (mu1 - mu2).dot((mu1 - mu2).T)
    weight = w1 + w2
    return mu, cov, weight


def wbarycenter_merge(w1, mu1, cov1, w2, mu2, cov2):
    w11, w21 = w1 / (w1 + w2), w2 / (w1 + w2)
    mu = w11 * mu1 + w21 * mu2
    cov = w11**2 * cov1 + w21**2 * cov2 + w11 * w21 * (
        linalg.sqrtm(cov2.dot(cov1)) + linalg.sqrtm(cov1.dot(cov2)))
    weight = w1 + w2
    return mu, cov, weight


def bound_on_KL(w1, cov1, w2, cov2, merged_cov):
    d = 0.5 * ((w1 + w2) * np.sum(np.log(linalg.eigvals(merged_cov))) -
               w1 * np.sum(np.log(linalg.eigvals(cov1))) -
               w2 * np.sum(np.log(linalg.eigvals(cov2))))
    return d


"""
Greedy algorithm for Gaussian mixture reduction
"""
def GMR_greedy(means, covs, weights, n_components, method="Salmond"):
    """Find a GMM with n_components that is closest
    to a GMM parameterized by means, covs, weights by greedy algorithm

    Parameters
    ----------
    means : numpy array, (N, d) 
    covs :  numpy array, (N, d, d) 
    weights: numpy array, (N, )
    n_components: integer>=1
    method: string: "Salmond", "Runnalls", "W", "Williams"

    Returns
    -------
    weights and support points of reduced GMM.
    """
    means = np.copy(means)
    covs = np.copy(covs)
    weights = np.copy(weights)
    N, d = means.shape
    M = n_components

    if method == "Salmond":
        # compute mean and covariance of the original mixture
        mu = np.sum(weights.reshape((-1, 1)) * means, axis=0)
        P = np.sum(weights.reshape((-1, 1, 1)) * covs, axis=0) + np.trace(
            np.diag(weights).dot((means - mu).dot((means - mu).T)))
        while N > M:
            distances = {}
            for i in range(N - 1):
                for j in range(i + 1, N):
                    delta_W = (weights[i] * weights[j]) / (
                        weights[i] + weights[j]) * (means[i] - means[j]).dot(
                            (means[i] - means[j]).T)
                    distances[(i, j)] = np.trace(np.linalg.inv(P).dot(delta_W))
            i, j = list(distances.keys())[np.array(list(
                distances.values())).argmin()]
            means[i], covs[i], weights[i] = moment_preserving_merge(
                weights[i], means[i], covs[i], weights[j], means[j], covs[j])
            means = np.delete(means, j, 0)
            covs = np.delete(covs, j, 0)
            weights = np.delete(weights, j)
            N -= 1

    elif method == "Runnalls" or method == "Williams":
        while N > M:
            distances = {}
            merged = {}
            for i in range(N - 1):
                for j in range(i + 1, N):
                    mu, cov, w = moment_preserving_merge(
                        weights[i], means[i], covs[i], weights[j], means[j],
                        covs[j])
                    merged[(i, j)] = [mu, cov, w]
                    if method == "Runnalls":
                        distances[(i,
                                   j)] = bound_on_KL(weights[i], covs[i],
                                                     weights[j], covs[j], cov)
                    elif method == "Williams":
                        distances[(i, j)] = GMM_L2(
                            [means[[i, j]], mu.reshape(1, d)],
                            [covs[[i, j]], cov.reshape(1, d, d)],
                            [weights[[i, j]], w.reshape(-1, )])
            i, j = list(distances.keys())[np.array(list(
                distances.values())).argmin()]
            means[i], covs[i], weights[i] = merged[(i, j)]
            means = np.delete(means, j, 0)
            covs = np.delete(covs, j, 0)
            weights = np.delete(weights, j)
            N -= 1

    elif method == "W":
        while N > M:
            distances = {}
            for i in range(N - 1):
                for j in range(i + 1, N):
                    distances[(i,
                               j)] = Gaussian_distance(means[i], means[j],
                                                       covs[i], covs[j],
                                                       "W2")**2
            i, j = list(distances.keys())[np.array(list(
                distances.values())).argmin()]
            means[i], covs[i], weights[i] = wbarycenter_merge(
                weights[i], means[i], covs[i], weights[j], means[j], covs[j])
            means = np.delete(means, j, 0)
            covs = np.delete(covs, j, 0)
            weights = np.delete(weights, j)
            N -= 1
    else:
        raise ValueError("This method is not implemented!")
    return means.astype(float), covs.astype(float), weights.astype(float)

