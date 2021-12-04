import ot
import numpy as np
from scipy import linalg
from .utils import log_normal
from scipy.stats import multivariate_normal
"""
Distance functions
Part I: distance between Gaussians
Part II: distance between Gaussian mixtures
"""


def Gaussian_distance(mu1, mu2, Sigma1, Sigma2, which='W2'):
    """
    Compute distance between Gaussians.

    Parameters
    ----------
    mu1 : array-like, shape (d, )
    mu2 : array-like, shape (d, )
    Sigma1 :  array-like, shape (d, d)
    Sigma2 :  array-like, shape (d, d)
    which : string, 'KL', 'WKL', 'CS', 'W2' and others


    Returns
    -------
    2-Wasserstein distance between Gaussians.

    """
    # print(np.linalg.eigvals(Sigma1))
    # print(np.linalg.eigvals(Sigma2))

    if which == 'KL':
        d = mu1.shape[0]
        # cholesky decomposition
        Sigma2_chol = linalg.cholesky(Sigma2, lower=True)
        Sigma1_chol = linalg.cholesky(Sigma1, lower=True)

        precisions_chol = linalg.solve_triangular(Sigma2_chol,
                                                  np.eye(d),
                                                  lower=True)
        # b = precisions_chol.T @ precisions_chol
        log_det = 2 * (np.sum(np.log(np.diag(Sigma2_chol))) -
                       np.sum(np.log(np.diag(Sigma1_chol))))

        prod = precisions_chol @ Sigma1_chol
        trace = np.trace(prod.T @ prod)

        quadratic_term = precisions_chol.dot(mu2 - mu1)
        quadratic_term = np.sum(quadratic_term**2)

        return .5 * (log_det + trace + quadratic_term - d)

    elif which == 'W2':
        # 1 dimensional
        if mu1.shape[0] == 1 or mu2.shape[0] == 1:
            W2_squared = (mu1 - mu2)**2 + (np.sqrt(Sigma1) -
                                           np.sqrt(Sigma2))**2
            W2_squared = np.asscalar(W2_squared)
        # multi-dimensional
        else:
            sqrt_Sigma1 = linalg.sqrtm(Sigma1)
            Sigma = Sigma1 + Sigma2 - 2 * linalg.sqrtm(
                sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1)
            W2_squared = np.linalg.norm(mu1 - mu2)**2 + np.trace(
                Sigma) + 1e-13
        return np.sqrt(W2_squared)

    elif which == 'L2':
        det_first_two = np.linalg.det(4 * np.pi * Sigma1)**(
            -1 / 2) + np.linalg.det(4 * np.pi * Sigma2)**(-1 / 2)
        l2_squared = det_first_two - 2 * multivariate_normal.pdf(
            mu1, mu2, Sigma1 + Sigma2)
        return l2_squared

    elif which == 'CS':
        log_det1 = np.log(np.linalg.eigvals(4 * np.pi * Sigma1)).sum()
        log_det2 = np.log(np.linalg.eigvals(4 * np.pi * Sigma2)).sum()

        return -multivariate_normal.logpdf(
            mu1, mu2, Sigma1 + Sigma2) - 0.25 * (log_det1 + log_det2)

        # return -np.log(multivariate_normal.pdf(
        #     mu1, mu2, Sigma1 + Sigma2)) + .5 * (
        #         np.log(multivariate_normal.pdf(mu1, mu1, 2 * Sigma1)) +
        #         np.log(multivariate_normal.pdf(mu2, mu2, 2 * Sigma2)))
        # return -np.log(
        #     multivariate_normal.pdf(mu1, mu2, Sigma1 + Sigma2) / np.sqrt(
        #         multivariate_normal.pdf(mu1, mu1, 2 * Sigma1) *
        #         multivariate_normal.pdf(mu2, mu2, 2 * Sigma2)))

    elif which == 'Hellinger':
        return

    else:
        raise ValueError('This ground distance is not implemented!')


def GMM_CTD(means,
            covs,
            weights,
            ground_distance='W2',
            matrix=False,
            N=1):
    """Compute the 2 Wasserstein distance between Gaussian mixtures.

    Parameters
    ----------
    means : list of numpy arrays, length 2, (k1,d), (k2,d)
    covs :  list of numpy arrays , length 2, (k1, d, d), (k2, d, d)
    weights: list of numpy arrays 
    Returns
    -------
    Composite Wasserstein distance.
    """

    mus1, mus2 = means[0], means[1]
    Sigmas1, Sigmas2 = covs[0], covs[1]

    k1, k2 = mus1.shape[0], mus2.shape[0]
    cost_matrix = np.zeros((k1, k2))

    w1, w2 = weights[0], weights[1]
    # w1 /= w1.sum()
    # w2 /= w2.sum()

    if ground_distance == 'KL':
        d = mus1.shape[1]

        diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]
        cost_matrix, precisions = log_normal(diff,
                                             np.tile(
                                                 Sigmas2,
                                                 (k1, 1, 1, 1)),
                                             prec=True)
        # corresponding to log\phi(\mu_1|mu_2,\sigma_2)
        # precision matrices
        precisions = precisions.reshape((k1, k2, d, d))
        traces = np.einsum('ijkl,ikl->ij', precisions, Sigmas1)

        log_det = np.zeros(k1)
        for i in range(k1):
            log_det[i] = np.sum(
                np.log(np.linalg.eigvals(2 * np.pi * Sigmas1[i])))

        cost_matrix = 0.5 * (traces.T - log_det - d).T - cost_matrix

    elif ground_distance == 'WKL':
        d = mus1.shape[1]
        diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]
        cost_matrix, precisions = log_normal(diff,
                                             np.tile(
                                                 Sigmas2,
                                                 (k1, 1, 1, 1)),
                                             prec=True)
        # precision matrices
        precisions = precisions.reshape((k1, k2, d, d))
        # precisions = precisions[]
        traces = np.einsum('ijkl,ikl->ij', precisions, Sigmas1)

        cost_matrix -= 0.5 * traces
        cost_matrix = -N * cost_matrix - np.log(w2)

    elif ground_distance == 'W2':
        for i in range(k1):
            for j in range(k2):
                cost_matrix[i, j] = Gaussian_distance(
                    mus1[i], mus2[j], Sigmas1[i], Sigmas2[j], 'W2')**2
    elif ground_distance == 'W1':
        for i in range(k1):
            for j in range(k2):
                cost_matrix[i, j] = np.linalg.norm(
                    mus1[i] - mus2[j]) + np.linalg.norm(
                        linalg.sqrtm(Sigmas1[i]) -
                        linalg.sqrtm(Sigmas2[j]))

    elif ground_distance == 'ISE':
        diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]
        covs = Sigmas2[np.newaxis, :] + Sigmas1[:, np.newaxis]
        cost_matrix = -2 * np.exp(log_normal(diff, covs))

        # add determinant
        col_det = np.zeros(k2)
        for i in range(k2):
            col_det[i] = np.linalg.det(
                4 * np.pi * Sigmas2[[i]])**(-.5)

        row_det = np.zeros(k1)
        for i in range(k1):
            row_det[i] = np.linalg.det(
                4 * np.pi * Sigmas1[[i]])**(-.5)

        cost_matrix += col_det
        cost_matrix = (cost_matrix.T + row_det).T

    elif ground_distance == 'WISE':
        diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]
        covs = Sigmas2[np.newaxis, :] + Sigmas1[:, np.newaxis]
        cost_matrix = -2 * np.outer(w1, w2) * np.exp(
            log_normal(diff, covs))

        # add determinant
        col_det = np.zeros(k2)
        for i in range(k2):
            col_det[i] = w2[i]**2 * np.linalg.det(
                4 * np.pi * Sigmas2[[i]])**(-.5)

        row_det = np.zeros(k1)
        for i in range(k1):
            row_det[i] = w1[i]**2 * np.linalg.det(
                4 * np.pi * Sigmas1[[i]])**(-.5)

        cost_matrix += col_det
        cost_matrix = (cost_matrix.T + row_det).T

    elif ground_distance == 'CS':
        d = mus1.shape[1]
        diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]

        covs = Sigmas2[np.newaxis, :] + Sigmas1[:, np.newaxis]
        cost_matrix = -log_normal(diff, covs)  # shape (k1, k2)
        # for each row, add the determinant of the covariance matrix of Sigmas1
        Sigmas1_log_det = np.zeros((k1, ))
        for i, cov in enumerate(Sigmas1):
            Sigmas1_log_det[i] = np.sum(np.log(
                np.linalg.eigvals(cov)))
        # for each column, add the determinant of the covariance matrix of Sigmas2
        Sigmas2_log_det = np.zeros((k2, ))
        for i, cov in enumerate(Sigmas2):
            Sigmas2_log_det[i] = np.sum(np.log(
                np.linalg.eigvals(cov)))
        cost_matrix = cost_matrix - (Sigmas1_log_det[:, np.newaxis] +
                                     Sigmas2_log_det[np.newaxis, :]
                                     ) / 4 - d / 2 * np.log(4 * np.pi)

    else:
        raise ValueError('This ground distance is not implemented!')

    if matrix:
        return cost_matrix
    else:
        CTD = ot.emd2(w1, w2, cost_matrix)
        return CTD


def GMM_L2(means, covs, weights, normalized=False):
    # compute the squared L2 distance between two mixtures
    w1, w2 = weights[0], weights[1]
    mus1, mus2 = means[0], means[1]
    Sigmas1, Sigmas2 = covs[0], covs[1]
    # normalization of the weights
    w1 /= w1.sum()
    w2 /= w2.sum()

    # S11
    diff = mus1[np.newaxis, :] - mus1[:, np.newaxis]
    covs = Sigmas1[np.newaxis, :] + Sigmas1[:, np.newaxis]
    S11 = np.exp(log_normal(diff, covs))

    # S12
    diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]
    covs = Sigmas2[np.newaxis, :] + Sigmas1[:, np.newaxis]
    S12 = np.exp(log_normal(diff, covs))

    # S22
    diff = mus2[np.newaxis, :] - mus2[:, np.newaxis]
    covs = Sigmas2[np.newaxis, :] + Sigmas2[:, np.newaxis]
    S22 = np.exp(log_normal(diff, covs))

    if normalized:
        return 1 - 2 * w1.T.dot(S12).dot(w2) / (
            w1.T.dot(S11).dot(w1) + w2.T.dot(S22).dot(w2))

    else:
        return w1.T.dot(S11).dot(
            w1) - 2 * w1.T.dot(S12).dot(w2) + w2.T.dot(S22).dot(w2)


def GMM_CS(means, covs, weights):
    #compute the Cauchy-Schwartz divergence between two mixtures
    w1, w2 = weights[0], weights[1]
    mus1, mus2 = means[0], means[1]
    Sigmas1, Sigmas2 = covs[0], covs[1]
    # normalization of the weights
    w1 /= w1.sum()
    w2 /= w2.sum()

    # S11
    diff = mus1[np.newaxis, :] - mus1[:, np.newaxis]
    covs = Sigmas1[np.newaxis, :] + Sigmas1[:, np.newaxis]
    S11 = np.exp(log_normal(diff, covs))

    # S12
    diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]
    covs = Sigmas2[np.newaxis, :] + Sigmas1[:, np.newaxis]
    # print(diff.shape, covs.shape)
    S12 = np.exp(log_normal(diff, covs))

    # S22
    diff = mus2[np.newaxis, :] - mus2[:, np.newaxis]
    covs = Sigmas2[np.newaxis, :] + Sigmas2[:, np.newaxis]
    S22 = np.exp(log_normal(diff, covs))
    return -np.log(w1.T.dot(S12).dot(w2)) + .5 * (
        np.log(w1.T.dot(S11).dot(w1)) + np.log(w2.T.dot(S22).dot(w2)))


# sanity check
if __name__ == '__main__':
    # d = 3
    # means = [
    #     np.random.randn(4, d),
    #     np.random.randn(3, d)
    # ]
    # covs = [
    #     np.empty((4, d, d)),
    #     np.empty((3, d, d))
    # ]
    # for i in range(4):
    #     a = np.random.randn(d, d)
    #     covs[0][i] = a @ a.T + 0.5 * np.eye(d)
    # for i in range(3):
    #     a = np.random.randn(d, d)
    #     covs[1][i] = a @ a.T + 0.5 * np.eye(d)

    # weights = [
    #     np.random.rand(4),
    #     np.random.rand(3)
    # ]
    means = [
        np.array([2.0, 2.0]).reshape((-1, 1)),
        np.array([2.0, 2.0]).reshape((-1, 1))
    ]

    covs = [
        np.array([1.0, 2.0]).reshape((-1, 1, 1)),
        np.array([1.0, 2.0]).reshape((-1, 1, 1))
    ]

    weights = [np.array([0.6, 0.4]), np.array([0.5, 0.5])]

    print(GMM_L2(means, covs, weights))
