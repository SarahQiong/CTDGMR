import numpy as np
from scipy import linalg
from scipy import optimize
from .distance import Gaussian_distance
from .utils import log_normal


def barycenter(means,
               covs,
               lambdas=None,
               tol=1e-7,
               ground_distance='W2'):
    """Compute the barycenter of Gaussian measures.

    Parameters
    ----------
    means : array-like, shape (n, d)
    covs :  array-like, shape (n, d, d)
    lambdas : array-like, shape (n,), weight in barycenter
    ground_distance: string. Options: "W2", "KL", "WKL" ,"Cauchy-Schwartz", "ISE"

    Returns
    -------
    mean and covariance of the Gaussian Wasserstein barycenter.

    """
    m, d = means.shape
    if lambdas is None:
        lambdas = np.ones((m, )) / m
    else:
        lambdas = lambdas / lambdas.sum()
        # weight normalization

    if ground_distance == 'KL' or ground_distance == 'WKL':
        barycenter_mean = np.sum((lambdas * means.T).T, axis=0)
        barycenter_cov = np.sum(covs * lambdas.reshape((-1, 1, 1)),
                                axis=0)
        diff = means - barycenter_mean
        barycenter_cov += np.dot(lambdas * diff.T, diff)
    elif ground_distance == 'W2':
        barycenter_mean = np.sum((lambdas * means.T).T, axis=0)
        if d == 1:
            barycenter_cov = np.sum(
                np.sqrt(covs) * lambdas.reshape((-1, 1, 1)))**2
        else:
            #Fixed point iteration for Gaussian barycenter
            barycenter_cov = barycenter(means,
                                        covs,
                                        lambdas,
                                        ground_distance='KL')[1]
            barycenter_cov_next = np.identity(d)
            while np.linalg.norm(barycenter_cov_next - barycenter_cov,
                                 'fro') > tol:
                barycenter_cov = barycenter_cov_next
                sqrt_barycenter_cov = linalg.sqrtm(barycenter_cov)
                barycenter_cov_next = np.zeros((d, d))
                for k in range(m):
                    barycenter_cov_next = barycenter_cov_next + lambdas[
                        k] * linalg.sqrtm(sqrt_barycenter_cov @ covs[k]
                                          @ sqrt_barycenter_cov)
    elif ground_distance == 'CS':
        # find the barycenter w.r.t. Cauchy-Schwartz divergence
        # using fixed point iteration
        def compute_sigma(covs, mus, cov, lambdas):
            # find (Sigma_r+Sigma)^{-1}
            covs = covs + cov
            for i, cov in enumerate(covs):
                covs[i] = np.linalg.inv(cov)

            # find (Sigma_r+Sigma)^{-1}(mu_r-mu)
            mu = compute_mean(covs, mus, cov, lambdas)
            mus = mus - mu
            weighted_mus = np.einsum('ijk,ik->ij', covs, mus)
            sandwich = np.einsum('ij,ik->ijk', weighted_mus,
                                 weighted_mus)
            return mu, 2 * ((covs - sandwich) *
                            lambdas[:, np.newaxis, np.newaxis]).sum(0)

        def compute_mean(precisions, mus, cov, lambdas):
            # precisions are: (Sigma_r+Sigma)^{-1}
            # find sum_{r}lambda_r(Sigma_r+Sigma)^{-1}
            weighted_precisions = precisions * lambdas[:, np.newaxis,
                                                       np.newaxis]
            # find sum_{r}lambda_r(Sigma_r+Sigma)^{-1}mu_r
            weighted_mus = np.einsum('ijk,ik->ij',
                                     weighted_precisions, mus)
            weighted_mus = weighted_mus.sum(0)
            return np.linalg.solve(weighted_precisions.sum(0),
                                   weighted_mus)

        # initial value for fixed point iteration
        barycenter_mean, barycenter_cov = barycenter(
            means, covs, lambdas, ground_distance='KL')
        barycenter_next = compute_sigma(covs, means, barycenter_cov,
                                        lambdas)
        barycenter_cov_next = np.linalg.inv(barycenter_next[1])
        n_iter = 0
        while np.linalg.norm(barycenter_cov_next - barycenter_cov,
                             'fro') > tol:
            n_iter += 1
            barycenter_cov = barycenter_cov_next
            barycenter_next = compute_sigma(covs, means,
                                            barycenter_cov, lambdas)
            barycenter_cov_next = np.linalg.inv(barycenter_next[1])
        barycenter_mean = barycenter_next[0]

    elif ground_distance == 'ISE':

        def obj(par, means, covs, lambdas):
            """
            par: shape (d+ d^2)
            means: shape (N, d)
            covs: shape (N, d, d)
            lambdas: shape (N,)

            Outputs:
            sum_{n} lambdas[n]*(|4*pi*Sigma|^{-1/2} 
            - 2 phi(mu| means[n], Sigma + covs[n]))
            """
            # standardize the weights
            # lambdas /= lambdas.sum()

            n, d = means.shape
            mean, cov_chol = par[:d], par[d:].reshape((d, d))
            cov = cov_chol.dot(cov_chol.T)

            if np.iscomplex(np.linalg.eigvals(cov_chol)).sum() > 0:
                return np.Inf
            else:
                diff = means - mean  # shape (N, d)
                covs = covs + cov  # shape (N, d, d)

                precisions_chol = np.zeros_like(covs)
                for k, sigma in enumerate(covs.reshape((-1, d, d))):
                    try:
                        sigma_chol = linalg.cholesky(sigma,
                                                     lower=True)
                    except linalg.LinAlgError:
                        raise ValueError('covariance chol is wrong.')
                    precisions_chol[k] = linalg.solve_triangular(
                        sigma_chol, np.eye(d), lower=True).T
                log_det = (np.sum(
                    np.log(
                        precisions_chol.reshape(n, -1)[:, ::d + 1]),
                    1))
                y = np.einsum('ij,ijk->ik', diff, precisions_chol)
                log_prob = np.sum(np.square(y), axis=1)
                log_probs = -.5 * (d * np.log(2 * np.pi) +
                                   log_prob) + log_det
                probs = np.exp(log_probs)
                return np.sum(
                    lambdas *
                    ((4 * np.pi)**(-d / 2) /
                     np.linalg.eigvals(cov_chol).prod() - 2 * probs))

        def grad(par, means, covs, lambdas):
            n, d = means.shape
            mean, cov_chol = par[:d], par[d:].reshape((d, d))
            cov = cov_chol.dot(cov_chol.T)
            if np.iscomplex(np.linalg.eigvals(cov_chol)).sum() > 0:
                return 1e8 * np.ones(d + d**2)
            else:
                diff = means - mean  # shape (N, d)
                covs = covs + cov  # shape (N, d, d)

                precisions_chol = np.zeros_like(covs)
                for k, sigma in enumerate(covs.reshape((-1, d, d))):
                    try:
                        sigma_chol = linalg.cholesky(sigma,
                                                     lower=True)
                    except linalg.LinAlgError:
                        raise ValueError('covariance chol is wrong.')
                    precisions_chol[k] = linalg.solve_triangular(
                        sigma_chol, np.eye(d), lower=True).T
                log_det = (np.sum(
                    np.log(
                        precisions_chol.reshape(n, -1)[:, ::d + 1]),
                    1))
                y = np.einsum('ij,ijk->ik', diff, precisions_chol)
                log_prob = np.sum(np.square(y), axis=1)
                log_probs = -.5 * (d * np.log(2 * np.pi) +
                                   log_prob) + log_det
                probs = np.exp(log_probs)
                precisions = np.stack([
                    prec_chol.dot(prec_chol.T)
                    for prec_chol in precisions_chol
                ])

                # partial derivative w.r.t. mean
                diff_std = np.einsum('ijk,ik->ij', precisions, diff)
                weighted_probs = probs * lambdas
                dLdmu = (diff_std.T * weighted_probs).T
                dLdmu = -2 * np.sum(dLdmu, 0)

                # partial derivative w.r.t. covariance
                sandwich = np.einsum('ij,ik->ijk', diff_std, diff_std)
                sandwich -= precisions
                dLdSigma = -2 * np.sum(
                    sandwich * weighted_probs.reshape((-1, 1, 1)), 0)
                dLdSigma = dLdSigma.dot(cov_chol)

                prec_chol = linalg.solve_triangular(cov_chol,
                                                    np.eye(d),
                                                    lower=True).T
                dLdSigma -= np.sum(lambdas) * (4 * np.pi)**(
                    -d / 2
                ) / np.linalg.eigvals(cov_chol).prod() * prec_chol

                return np.concatenate((dLdmu, dLdSigma.reshape(
                    (-1, ))))

        obj_lambda = lambda x: obj(x, means, covs, lambdas)
        grad_lambda = lambda x: grad(x, means, covs, lambdas)

        barycenter_mean, barycenter_cov = barycenter(
            means, covs, lambdas, ground_distance='KL')
        barycenter_cholesky = linalg.cholesky(barycenter_cov,
                                              lower=True)
        x0 = np.concatenate(
            (barycenter_mean, barycenter_cholesky.reshape((-1, ))))
        res = optimize.minimize(obj_lambda,
                                x0,
                                method='BFGS',
                                jac=grad_lambda)
        if not res.success:
            barycenter_mean, barycenter_cov = barycenter(
                means, covs, lambdas, ground_distance='KL')
            barycenter_cholesky = linalg.cholesky(barycenter_cov,
                                                  lower=True)
            x0 = np.concatenate(
                (barycenter_mean, barycenter_cholesky.reshape(
                    (-1, ))))
            res = optimize.minimize(obj_lambda,
                                    x0,
                                    method='Nelder-Mead')
        if res.success:
            barycenter_mean = res.x[:d]
            barycenter_chol = res.x[d:].reshape((d, d))
            barycenter_cov = barycenter_chol.dot(barycenter_chol.T)
        else:
            print(res)
    else:
        raise ValueError(
            'This ground_distance %s is no implemented.' %
            ground_distance)

    return barycenter_mean, barycenter_cov


# sanity check
if __name__ == '__main__':
    d = 3
    means = np.random.randn(4, d)
    covs = np.empty((4, d, d))
    for i in range(4):
        a = np.random.randn(d, d)
        covs[i] = a @ a.T + 0.5 * np.eye(d)
        # print(np.linalg.eigvals(covs[i]))
    weights = np.ones(4) / 4

    barycenter_mean, barycenter_cov = barycenter(means,
                                                 covs,
                                                 weights,
                                                 ground_distance='KL')
    print(barycenter_mean, barycenter_cov)

    barycenter_mean, barycenter_cov = barycenter(means,
                                                 covs,
                                                 weights,
                                                 ground_distance='L2',
                                                 coeffs=np.array(
                                                     [1, 1]))
    print(barycenter_mean, barycenter_cov)
