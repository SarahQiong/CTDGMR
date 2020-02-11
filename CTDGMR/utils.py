import ot
import numpy as np
from scipy import linalg
import scipy.stats as ss
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture.base import _check_shape
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state


def _check_weights(weights):
    # check range
    if (any(np.less(weights, 0.)) or any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got min value %.5f, max value %.5f" %
                         (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))


def log_normal_pdf(x, mean, cov):
    d = x.shape[0]
    # cholesky decomposition of precision matrix
    cov_chol = linalg.cholesky(cov, lower=True)
    prec_chol = linalg.solve_triangular(cov_chol, np.eye(d), lower=True).T
    # log determinant of cholesky matrix
    log_det = np.sum(np.log(np.diag(prec_chol)))
    y = np.dot(x - mean, prec_chol)
    log_prob = np.sum(np.square(y))
    return -.5 * (d * np.log(2 * np.pi) + log_prob) + log_det


def fEij(mean, cov, reduced_mean, reduced_cov):
    return log_normal_pdf(mean, reduced_mean, reduced_cov) - 0.5 * np.trace(
        linalg.inv(reduced_cov) @ cov)


def Elogp(means, covs, reduced_means, reduced_covs):
    n, m = means.shape[0], reduced_means.shape[0]
    Eij = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            Eij[i, j] = fEij(means[i], covs[i], reduced_means[j],
                             reduced_covs[j])
    return Eij

def compute_precision_cholesky(covariances):
    """Compute the Cholesky decomposition of the precisions"""
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                     np.eye(n_features),
                                                     lower=True).T
    return precisions_chol


def compute_log_det_cholesky(matrix_chol, n_features):
    n_components, _, _ = matrix_chol.shape
    log_det_chol = (np.sum(np.log(
        matrix_chol.reshape(
            n_components, -1)[:, ::n_features + 1]), 1))
    return log_det_chol
    

def compute_resp(X, means, covs):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    precisions_chol = compute_precision_cholesky(covs)
    log_det = compute_log_det_cholesky(precisions_chol, n_features)
    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)
    log_resp = -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
    return log_resp



def log_GMM_pdf(X, means, covs, weights):
    resp = compute_resp(X, means, covs)
    return np.log(np.sum(np.exp(resp) * weights, axis=1))

    
def MC_KL(means, covs, weights):
    # Monte Carlo estimation for KL divergence from original to reduced mixture
    base_means, r_means = means[0], means[1]
    base_covs, r_covs = covs[0], covs[1]
    base_weights, r_weights = weights[0], weights[1]

    n_samples = 10000
    X, _ = GMM_sampler(base_means, base_covs, base_weights, n_samples)
    logp = log_GMM_pdf(X, base_means, base_covs, base_weights)
    logq = log_GMM_pdf(X, r_means, r_covs, r_weights)
    
    return np.mean(logp - logq)

def MC_ll(means, covs, weights):
    # Monte Carlo estimation for KL divergence from original to reduced mixture
    base_means, r_means = means[0], means[1]
    base_covs, r_covs = covs[0], covs[1]
    base_weights, r_weights = weights[0], weights[1]

    n_samples = 10000
    X, _ = GMM_sampler(base_means, base_covs, base_weights, n_samples)
    logq = log_GMM_pdf(X, r_means, r_covs, r_weights)
    
    return np.mean(logq)

def GMM_sampler(means, covs, weights, n_samples, random_state=0):
    """Sample from a Gaussian mixture

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
    weights = check_array(weights,
                          dtype=[np.float64, np.float32],
                          ensure_2d=False)
    # check range
    if (any(np.less(weights, 0.)) or any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f" %
                         (np.min(weights), np.max(weights)))
    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
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


"""
Distance functions

Part I: distance between Gaussians
Part II: distance between Gaussian mixtures

"""


def Gaussian_distance(mu1, mu2, Sigma1, Sigma2, which="W2"):
    """Compute distance between Gaussians.

    Parameters
    ----------
    mu1 : array-like, shape (d, )
    mu2 : array-like, shape (d, )
    Sigma1 :  array-like, shape (d, d)
    Sigma2 :  array-like, shape (d, d)
    which : string, "W2" or "KL"


    Returns
    -------
    2-Wasserstein distance between Gaussians.

    """
    if which == "KL":
        d = mu1.shape[0]
        # Sigma2_chol = np.linalg.cholesky(Sigma2)
        # log_det = 2 * np.log(np.linalg.det(Sigma2_chol)) - np.log(
        #     np.linalg.det(Sigma1))
        # trace = np.trace(linalg.cho_solve((Sigma2_chol, True), Sigma1))
        # chol_mu = np.dot(linalg.inv(Sigma2_chol), (mu2 - mu1))
        # quadratic_term = chol_mu.T.dot(chol_mu)
        Sigma2_inv = np.linalg.inv(Sigma2)
        log_det = -(np.log(np.linalg.det(Sigma2_inv)) +
                    np.log(np.linalg.det(Sigma1)))
        trace = np.matrix.trace(Sigma2_inv.dot(Sigma1))
        quadratic_term = (mu2 - mu1).T.dot(Sigma2_inv).dot(mu2 - mu1)
        return .5 * (log_det + trace + quadratic_term - d)

    elif which == "W2":
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
            W2_squared = np.linalg.norm(mu1 - mu2)**2 + np.trace(Sigma)
        return np.sqrt(W2_squared)
    else:
        raise ValueError("This ground distance is not implemented!")


def log_normals(diffs, covs):
    """
    log normal density of a matrix X
    evaluated for multiple multivariate normals
    =====
    input:
    diffs: array-like (N, M, d)
    covs: array-like (N,M,d,d)
    """
    n, m, d, _ = covs.shape
    if d == 1:
        precisions_chol = (np.sqrt(1/covs)).reshape((n*m,d,d))
    else:
        precisions_chol = np.empty((n * m, d, d))
        for k, cov in enumerate(covs.reshape((-1, d, d))):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                raise ValueError("Precision chol is wrong.")
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(d),
                                                         lower=True).T
    log_det = (np.sum(np.log(precisions_chol.reshape(n * m, -1)[:, ::d + 1]),
                      1))
    diffs = diffs.reshape((-1, d))
    y = np.einsum('ij,ijk->ik', diffs, precisions_chol)
    log_prob = np.sum(np.square(y), axis=1)
    probs = -.5 * (d * np.log(2 * np.pi) + log_prob) + log_det
    return probs.reshape((n, m))


def GMM_L2(means, covs, weights, normalized=False):
    """Compute the squared L2 distance between Gaussian mixtures.

    Parameters
    ----------
    means : list of numpy arrays, length 2, (k1,d), (k2,d)
    covs :  list of numpy arrays , length 2, (k1, d, d), (k2, d, d)
    weights: list of numpy arrays 
    Returns
    -------
    Squared L2 distance between Gaussian mixtures.
    """
    w1, w2 = weights[0], weights[1]
    mus1, mus2 = means[0], means[1]
    Sigmas1, Sigmas2 = covs[0], covs[1]
    # normalization of the weights
    w1 /= w1.sum()
    w2 /= w2.sum()

    # M11
    diff = mus1[np.newaxis, :] - mus1[:, np.newaxis]
    covs = Sigmas1[np.newaxis, :] + Sigmas1[:, np.newaxis]
    M11 = np.exp(log_normals(diff, covs))

    # M12
    diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]
    covs = Sigmas2[np.newaxis, :] + Sigmas1[:, np.newaxis]
    M12 = np.exp(log_normals(diff, covs))

    # M22
    diff = mus2[np.newaxis, :] - mus2[:, np.newaxis]
    covs = Sigmas2[np.newaxis, :] + Sigmas2[:, np.newaxis]
    M22 = np.exp(log_normals(diff, covs))

    return w1.T.dot(M11).dot(w1) - 2 * w1.T.dot(M12).dot(w2) + w2.T.dot(
        M22).dot(w2)


def GMM_CWD(means, covs, weights=None, ground_distance="W2", matrix=False,
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
    w1 /= w1.sum()
    w2 /= w2.sum()
    _check_weights(w1)
    _check_weights(w2)

    for i in range(k1):
        for j in range(k2):
            if ground_distance == "W2":
                cost_matrix[i, j] = Gaussian_distance(mus1[i], mus2[j],
                                                      Sigmas1[i], Sigmas2[j],
                                                      "W2")**2
            elif ground_distance == "KL":
                cost_matrix[i, j] = Gaussian_distance(mus1[i], mus2[j],
                                                      Sigmas1[i], Sigmas2[j],
                                                      "KL")
            elif ground_distance == "WKL":
                cost_matrix[i, j] = -(np.log(w2[j]) + N * fEij(
                    mus1[i], Sigmas1[i], mus2[j], Sigmas2[j]))

            elif ground_distance == "W1":
                cost_matrix[i, j] = np.linalg.norm(
                    mus1[i] - mus2[j]) + np.linalg.norm(
                        linalg.sqrtm(Sigmas1[i]) - linalg.sqrtm(Sigmas2[j]))
            else:
                raise ValueError("This ground distance is not implemented!")

    if matrix:
        return cost_matrix
    else:
        GMM_Wdistance = ot.emd2(w1, w2, cost_matrix)
        if ground_distance == "W2":
            return np.sqrt(GMM_Wdistance)
        else:
            return GMM_Wdistance


def Gaussian_barycenter(means,
                        covs,
                        weights=None,
                        tol=1e-5,
                        ground_distance="W2"):
    """Compute the Wasserstein or KL barycenter of Gaussian measures.

    Parameters
    ----------
    means : array-like, shape (n, d)
    covs :  array-like, shape (n, d, d)
    weights : array-like, shape (n,)
        The weight in front of the Wasserstein distance.
    ground_distance: string. One of "W2" and "KL"

    Returns
    -------
    mean and covariance of the Gaussian Wasserstein barycenter.

    """
    m, d = means.shape
    if weights is None:
        weights = np.ones((m, 1)) / m
    else:
        # weight standardization
        weights = weights / weights.sum()
        weights = check_array(weights,
                              dtype=[np.float64, np.float32],
                              ensure_2d=False)
        _check_shape(weights, (m, ), 'weights')
        # check range
        if (any(np.less(weights, 0.)) or any(np.greater(weights, 1.))):
            raise ValueError("The parameter 'weights' should be in the range "
                             "[0, 1], but got max value %.5f, min value %.5f" %
                             (np.min(weights), np.max(weights)))
        # check normalization
        if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
            raise ValueError("The parameter 'weights' should be normalized, "
                             "but got sum(weights) = %.5f" % np.sum(weights))

    barycenter_means = np.sum(weights.reshape((-1, 1)) * means, axis=0)

    if ground_distance == "KL" or ground_distance == "WKL":
        barycenter_covs = np.sum(covs * weights.reshape((-1, 1, 1)), axis=0)
        diff = means - barycenter_means
        barycenter_covs += np.dot(weights * diff.T, diff)

    elif ground_distance == "W2":
        if d == 1:
            barycenter_covs = np.sum(
                np.sqrt(covs) * weights.reshape((-1, 1, 1)))**2
        else:
            #Fixed point iteration for Gaussian barycenter
            barycenter_covs = np.zeros((d, d))
            barycenter_covs_next = np.identity(d)
            while np.linalg.norm(barycenter_covs_next - barycenter_covs,
                                 'fro') > tol:
                barycenter_covs = barycenter_covs_next
                sqrt_barycenter_covs = linalg.sqrtm(barycenter_covs)
                barycenter_covs_next = np.zeros((d, d))
                for k in range(m):
                    barycenter_covs_next = barycenter_covs_next + \
                    weights[k] * linalg.sqrtm(sqrt_barycenter_covs@covs[k]@sqrt_barycenter_covs)

    else:
        raise ValueError("This ground_distance %s is no implemented." %
                         ground_distance)
    return barycenter_means, barycenter_covs


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


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=False, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='rainbow', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def gmm_plot2d(means, covs, weights):
    x = np.linspace(-15, 15, 100)
    y = np.linspace(-15, 15, 100)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros(xx.shape)
    pos = np.empty(xx.shape + (2, ))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    for (l, s), w in zip(zip(means, covs), weights):
        zz += ss.multivariate_normal.pdf(pos, mean=l, cov=s) * w
    plt.imshow(zz, extent=[np.min(x), np.max(x), np.min(y), np.max(y)])


def gmm_plot1d(gmm, color, opt='cpg', dim=0):
    """plot a gmm in 1d
    INPUTS:
      opt = 'c' -- plot component
          = 'p' -- plot priors
          = 'g' -- plot gmm
      dim = dimension to use [default = 0]"""

    # # select dimension (if necessary)
    # if len(gmm.means_[0]) != 1:
    #     for t in range(len(gmm.means_)):
    #         gmm.means_[t] = gmm.means_[t][dim]
    #         if gmm.covariance_type == 'diag':
    #             gmm.covariances_[t] = gmm.covariances_[t][dim]
    #         elif gmm.covariance_type == 'full':
    #             gmm.covariances_[t] = gmm.covariances_[t][dim, dim]

    stdext = 3
    xlo = []
    xhi = []
    for j in range(gmm.n_components):
        xlo.append(gmm.means_[j][0] -
                   np.sqrt(gmm.covariances_[j][0][0]) * stdext)
        xhi.append(gmm.means_[j][0] +
                   np.sqrt(gmm.covariances_[j][0][0]) * stdext)

    xlo = min(xlo)
    xhi = max(xhi)

    x = np.linspace(xlo, xhi, 200)
    x = x.reshape(-1, 1)

    LL, LLcomp, post = gmm_ll(x, gmm)
    # print LL
    for j in range(gmm.n_components):
        if 'c' in opt:
            plt.plot(x, np.exp(LLcomp[:, j]), color + ':')
        # if 'p' in opt:
        # text display the priors
        if 'g' in opt:
            plt.plot(x, np.exp(LL), color + '-')
