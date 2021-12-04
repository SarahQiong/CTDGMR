import ot
import time
import warnings
import numpy as np
from scipy import linalg
from scipy import optimize
from scipy.special import logsumexp, softmax
from scipy.stats import multivariate_normal
from numpy.linalg import det
from cvxopt import matrix
from cvxopt import solvers
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from .greedy import *
from .utils import *
from .distance import *


###########################################
# objective and gradients
###########################################
def opt_obj(reduced_means,
            reduced_covs_chol,
            reduced_weights,
            means,
            covs,
            weights,
            chol=True,
            loss='ISE'):
    if chol:
        reduced_covs = np.zeros_like(reduced_covs_chol)
        for i in range(reduced_means.shape[0]):
            reduced_covs[i] = reduced_covs_chol[i].dot(reduced_covs_chol[i].T)
    else:
        reduced_covs = reduced_covs_chol
    # compute the similarity matrices
    SRR_diff = reduced_means[np.newaxis, :] - reduced_means[:, np.newaxis]
    SRR_covs = reduced_covs[np.newaxis, :] + reduced_covs[:, np.newaxis]
    SRR = np.exp(log_normal(SRR_diff, SRR_covs))
    SOR_diff = reduced_means[np.newaxis, :] - means[:, np.newaxis]
    SOR_covs = reduced_covs[np.newaxis, :] + covs[:, np.newaxis]
    SOR = np.exp(log_normal(SOR_diff, SOR_covs))
    if loss == 'NISE':
        SOO_diff = means[np.newaxis, :] - means[:, np.newaxis]
        SOO_covs = covs[np.newaxis, :] + covs[:, np.newaxis]
        SOO = np.exp(log_normal(SOO_diff, SOO_covs))

    # return the objective functions
    if loss == 'CS':
        return -np.log(weights.T.dot(SOR).dot(reduced_weights)) + .5 * np.log(
            reduced_weights.T.dot(SRR).dot(reduced_weights))
    elif loss == 'ISE':
        return reduced_weights.T.dot(SRR).dot(
            reduced_weights) - 2 * weights.T.dot(SOR).dot(reduced_weights)

    elif loss == 'NISE':
        # we work with the logorithm version
        return -np.log(weights.T.dot(SOR).dot(reduced_weights)) + np.log(
            reduced_weights.T.dot(SRR).dot(reduced_weights) +
            weights.T.dot(SOO).dot(weights))


# gradients wrt to reduced model parameters
def obj_grads_theta(reduced_means,
                    reduced_covs_chol,
                    reduced_weights,
                    means,
                    covs,
                    weights,
                    loss='ISE'):
    """
    The gradient with respect to the subpopulation 
    means and choleskdy decomposition of covariance
    """
    reduced_covs = np.zeros_like(reduced_covs_chol)
    for i in range(reduced_means.shape[0]):
        reduced_covs[i] = np.dot(reduced_covs_chol[i], reduced_covs_chol[i].T)
    n = means.shape[0]
    m, d = reduced_means.shape

    # S12
    S12_diff = reduced_means[np.newaxis, :] - means[:, np.newaxis]
    S12_cov = reduced_covs[np.newaxis, :] + covs[:, np.newaxis]
    S12, S12_precision = log_normal(S12_diff, S12_cov, prec=True)
    S12 = np.exp(S12)
    # S12_precision = S12_precision.reshape((n, m, d, d))

    # S22
    S22_diff = reduced_means[np.newaxis, :] - reduced_means[:, np.newaxis]
    S22_cov = reduced_covs[np.newaxis, :] + reduced_covs[:, np.newaxis]
    S22, S22_precision = log_normal(S22_diff, S22_cov, prec=True)
    S22 = np.exp(S22)
    # S22_precision = S22_precision.reshape((m, m, d, d))

    # S11
    if loss == 'NISE':
        S11_diff = means[np.newaxis, :] - means[:, np.newaxis]
        S11_cov = covs[np.newaxis, :] + covs[:, np.newaxis]
        S11 = np.exp(log_normal(S11_diff, S11_cov))

    # gradient w.r.t. subpop means
    L12_mean_std = np.einsum('ijk,ik->ij', S12_precision,
                             S12_diff.reshape((-1, d)))
    weighted_S12 = S12 * weights[:,
                                 np.newaxis] * reduced_weights[np.newaxis, :]
    dL12dreduced_mean = L12_mean_std.reshape(
        (n, m, d)) * weighted_S12[:, :, np.newaxis]
    dL12dreduced_mean = -np.sum(dL12dreduced_mean, 0)

    L22_mean_std = np.einsum('ijk,ik->ij', S22_precision,
                             S22_diff.reshape((-1, d)))
    weighted_S22 = 2 * S22 * reduced_weights[:, np.newaxis] * reduced_weights[
        np.newaxis, :]
    dL22dreduced_mean = L22_mean_std.reshape(
        (m, m, d)) * weighted_S22[:, :, np.newaxis]
    dL22dreduced_mean = -np.sum(dL22dreduced_mean, 0)

    # gradient w.r.t. cholesky decomposition of subpop covariances
    sandwich = (np.einsum('ij,ik->ijk', L22_mean_std, L22_mean_std) -
                S22_precision).reshape(m, m, d, d)
    sandwich = sandwich * weighted_S22[:, :, np.newaxis, np.newaxis]
    dL22dreduced_cov_chol = np.sum(sandwich, 0)
    dL22dreduced_cov_chol = np.einsum('ikl,ils->iks', dL22dreduced_cov_chol,
                                      reduced_covs_chol)

    sandwich = (np.einsum('ij,ik->ijk', L12_mean_std, L12_mean_std) -
                S12_precision).reshape(n, m, d, d)
    sandwich = sandwich * weighted_S12[:, :, np.newaxis, np.newaxis]
    dL12dreduced_cov_chol = np.sum(sandwich, 0)
    dL12dreduced_cov_chol = np.einsum('ikl,ils->iks', dL12dreduced_cov_chol,
                                      reduced_covs_chol)

    if loss == 'ISE':
        grad_reduced_means = dL22dreduced_mean - 2 * dL12dreduced_mean
        grad_reduced_covs_chol = dL22dreduced_cov_chol - 2 * dL12dreduced_cov_chol
    elif loss == 'NISE':
        L11 = (weights.T).dot(S11).dot(weights)
        L12 = (weights.T).dot(S12).dot(reduced_weights)
        L22 = (reduced_weights.T).dot(S22).dot(reduced_weights)
        grad_reduced_means = dL22dreduced_mean / (
            L11 + L22) - 1 / L12 * dL12dreduced_mean
        grad_reduced_covs_chol = dL22dreduced_cov_chol / (
            L11 + L22) - 1 / L12 * dL12dreduced_cov_chol

    elif loss == 'CS':
        L12 = (weights.T).dot(S12).dot(reduced_weights)
        L22 = (reduced_weights.T).dot(S22).dot(reduced_weights)
        grad_reduced_means = dL22dreduced_mean / (
            2 * L22) - 1 / L12 * dL12dreduced_mean
        grad_reduced_covs_chol = dL22dreduced_cov_chol / (
            2 * L22) - 1 / L12 * dL12dreduced_cov_chol

    return np.concatenate((grad_reduced_means.reshape(
        (-1, )), grad_reduced_covs_chol.reshape((-1, ))))


# gradients wrt to reduced model mixing weights
# this is only used for NISE and CS
def obj_grads_w(reduced_means,
                reduced_covs,
                reduced_weights,
                means,
                covs,
                weights,
                loss='ISE'):
    # S12
    S12_diff = reduced_means[np.newaxis, :] - means[:, np.newaxis]
    S12_cov = reduced_covs[np.newaxis, :] + covs[:, np.newaxis]
    S12 = log_normal(S12_diff, S12_cov)
    S12 = np.exp(S12)

    # S22
    S22_diff = reduced_means[np.newaxis, :] - reduced_means[:, np.newaxis]
    S22_cov = reduced_covs[np.newaxis, :] + reduced_covs[:, np.newaxis]
    S22 = log_normal(S22_diff, S22_cov)
    S22 = np.exp(S22)

    # S11
    if loss == 'NISE':
        S11_diff = means[np.newaxis, :] - means[:, np.newaxis]
        S11_cov = covs[np.newaxis, :] + covs[:, np.newaxis]
        S11 = np.exp(log_normal(S11_diff, S11_cov))

    dL12dw = weights.T.dot(S12)
    dL22dw = 2 * reduced_weights.T.dot(S22)
    # gradient w.r.t the unconstraint parameters
    dL12dt = dL12dw * reduced_weights - reduced_weights * np.sum(
        dL12dw * reduced_weights)
    dL22dt = dL22dw * reduced_weights - reduced_weights * np.sum(
        dL22dw * reduced_weights)

    if loss == 'ISE':
        return dL22dt - 2 * dL12dt
    elif loss == 'NISE':
        L11 = (weights.T).dot(S11).dot(weights)
        L12 = (weights.T).dot(S12).dot(reduced_weights)
        L22 = (reduced_weights.T).dot(S22).dot(reduced_weights)
        return dL22dt / (L11 + L22) - 1 / L12 * dL12dt
    elif loss == 'CS':
        L12 = (weights.T).dot(S12).dot(reduced_weights)
        L22 = (reduced_weights.T).dot(S22).dot(reduced_weights)
        return dL22dt / (2 * L22) - 1 / L12 * dL12dt


# gradients wrt to reduced model parameters
def obj_grads(reduced_means,
              reduced_covs_chol,
              reduced_weights,
              means,
              covs,
              weights,
              loss='ISE'):
    """
    The gradient with respect to the subpopulation 
    means and choleskdy decomposition of covariance
    """
    reduced_covs = np.zeros_like(reduced_covs_chol)
    for i in range(reduced_means.shape[0]):
        reduced_covs[i] = np.dot(reduced_covs_chol[i], reduced_covs_chol[i].T)
    n = means.shape[0]
    m, d = reduced_means.shape

    # S12
    S12_diff = reduced_means[np.newaxis, :] - means[:, np.newaxis]
    S12_cov = reduced_covs[np.newaxis, :] + covs[:, np.newaxis]
    S12, S12_precision = log_normal(S12_diff, S12_cov, prec=True)
    S12 = np.exp(S12)
    # S12_precision = S12_precision.reshape((n, m, d, d))

    # S22
    S22_diff = reduced_means[np.newaxis, :] - reduced_means[:, np.newaxis]
    S22_cov = reduced_covs[np.newaxis, :] + reduced_covs[:, np.newaxis]
    S22, S22_precision = log_normal(S22_diff, S22_cov, prec=True)
    S22 = np.exp(S22)
    # S22_precision = S22_precision.reshape((m, m, d, d))

    # S11
    if loss == 'NISE':
        S11_diff = means[np.newaxis, :] - means[:, np.newaxis]
        S11_cov = covs[np.newaxis, :] + covs[:, np.newaxis]
        S11 = np.exp(log_normal(S11_diff, S11_cov))

    # gradient w.r.t. subpop means
    L12_mean_std = np.einsum('ijk,ik->ij', S12_precision,
                             S12_diff.reshape((-1, d)))
    weighted_S12 = S12 * weights[:,
                                 np.newaxis] * reduced_weights[np.newaxis, :]
    dL12dreduced_mean = L12_mean_std.reshape(
        (n, m, d)) * weighted_S12[:, :, np.newaxis]
    dL12dreduced_mean = -np.sum(dL12dreduced_mean, 0)

    L22_mean_std = np.einsum('ijk,ik->ij', S22_precision,
                             S22_diff.reshape((-1, d)))
    weighted_S22 = 2 * S22 * reduced_weights[:, np.newaxis] * reduced_weights[
        np.newaxis, :]
    dL22dreduced_mean = L22_mean_std.reshape(
        (m, m, d)) * weighted_S22[:, :, np.newaxis]
    dL22dreduced_mean = -np.sum(dL22dreduced_mean, 0)

    # gradient w.r.t. cholesky decomposition of subpop covariances
    sandwich = (np.einsum('ij,ik->ijk', L22_mean_std, L22_mean_std) -
                S22_precision).reshape(m, m, d, d)
    sandwich = sandwich * weighted_S22[:, :, np.newaxis, np.newaxis]
    dL22dreduced_cov_chol = np.sum(sandwich, 0)
    dL22dreduced_cov_chol = np.einsum('ikl,ils->iks', dL22dreduced_cov_chol,
                                      reduced_covs_chol)

    sandwich = (np.einsum('ij,ik->ijk', L12_mean_std, L12_mean_std) -
                S12_precision).reshape(n, m, d, d)
    sandwich = sandwich * weighted_S12[:, :, np.newaxis, np.newaxis]
    dL12dreduced_cov_chol = np.sum(sandwich, 0)
    dL12dreduced_cov_chol = np.einsum('ikl,ils->iks', dL12dreduced_cov_chol,
                                      reduced_covs_chol)

    dL12dw = weights.T.dot(S12)
    dL22dw = 2 * reduced_weights.T.dot(S22)
    # gradient w.r.t the unconstraint parameters
    dL12dt = dL12dw * reduced_weights - reduced_weights * np.sum(
        dL12dw * reduced_weights)
    dL22dt = dL22dw * reduced_weights - reduced_weights * np.sum(
        dL22dw * reduced_weights)

    if loss == 'ISE':
        grad_reduced_means = dL22dreduced_mean - 2 * dL12dreduced_mean
        grad_reduced_covs_chol = dL22dreduced_cov_chol - 2 * dL12dreduced_cov_chol
        grad_reduced_weights = dL22dt - 2 * dL12dt
    elif loss == 'NISE':
        L11 = (weights.T).dot(S11).dot(weights)
        L12 = (weights.T).dot(S12).dot(reduced_weights)
        L22 = (reduced_weights.T).dot(S22).dot(reduced_weights)
        grad_reduced_means = dL22dreduced_mean / (
            L11 + L22) - 1 / L12 * dL12dreduced_mean
        grad_reduced_covs_chol = dL22dreduced_cov_chol / (
            L11 + L22) - 1 / L12 * dL12dreduced_cov_chol
        grad_reduced_weights = dL22dt / (L11 + L22) - 1 / L12 * dL12dt

    elif loss == 'CS':
        L12 = (weights.T).dot(S12).dot(reduced_weights)
        L22 = (reduced_weights.T).dot(S22).dot(reduced_weights)
        grad_reduced_means = dL22dreduced_mean / (
            2 * L22) - 1 / L12 * dL12dreduced_mean
        grad_reduced_covs_chol = dL22dreduced_cov_chol / (
            2 * L22) - 1 / L12 * dL12dreduced_cov_chol
        grad_reduced_weights = dL22dt / (2 * L22) - 1 / L12 * dL12dt

    return np.concatenate(
        (grad_reduced_weights, grad_reduced_means.reshape(
            (-1, )), grad_reduced_covs_chol.reshape((-1, ))))


##########################################
# optimization based method for reduction
##########################################
# this code implements the minimum ISE, NISE,
# Cauchy-Schwartz divergence between two mixtures for GMR


class GMR_opt:
    """
    Find a GMM with n_components that is closest
    to a GMM parameterized by means, covs, weights 
    by optimization based method.
    The distances implemented are ISE, NISE, and CS

    Parameters
    ----------
    means : numpy array, (N, d) 
    covs :  numpy array, (N, d, d) 
    weights: numpy array, (N, )
    Returns
    -------
    weights and support points of reduced GMM.
    """
    def __init__(self,
                 means,
                 covs,
                 weights,
                 n,
                 loss='ISE',
                 init_method='Runnalls',
                 tol=1e-5,
                 max_iter=100,
                 random_state=0,
                 reduced_means=None,
                 reduced_covs=None,
                 reduced_weights=None):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = n
        self.d = means.shape[1]
        self.random_state = random_state
        self.converged_ = False
        self.init_method = init_method
        self.loss = loss

    def _initialize_parameter(self):
        """Initializatin of the reduced mixture"""
        # self.H1 = np.zeros((self.new_n, self.new_n))
        # self.H2 = np.zeros((self.origin_n, self.new_n))
        # if self.loss == 'NISE':
        #     self.H3 = np.zeros((self.origin_n, self.origin_n))

        if self.init_method == 'kmeans':
            total_sample_size = 10000
            X = rmixGaussian(self.means, self.covs, self.weights,
                             total_sample_size, self.random_state)[0]
            gm = GaussianMixture(n_components=self.new_n,
                                 random_state=0).fit(X)
            self.reduced_means = gm.means_
            self.reduced_covs = gm.covariances_
            self.reduced_weights = gm.weights_
        elif self.init_method == 'user':
            self.reduced_means = reduced_means
            self.reduced_covs = reduced_covs
            self.reduced_weights = reduced_weights
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.new_n,
                self.init_method)
        # self._update_H1_H2()

    def _obj(self):
        if self.loss == 'CS':
            return -np.log(
                np.dot(self.weights.T, self.H2).dot(
                    self.reduced_weights)) + .5 * np.log(
                        np.dot(self.reduced_weights.T, self.H1).dot(
                            self.reduced_weights))
        elif self.loss == 'ISE':
            return np.dot(self.reduced_weights.T, self.H1).dot(
                self.reduced_weights) - 2 * np.dot(
                    self.weights.T, self.H2).dot(self.reduced_weights)

        elif self.loss == 'NISE':
            # we work with the logorithm version
            return -np.log(
                np.dot(self.weights.T, self.H2).dot(
                    self.reduced_weights)) + np.log(
                        np.dot(self.reduced_weights.T, self.H1).dot(
                            self.reduced_weights) +
                        np.dot(self.weights.T, self.H3).dot(self.weights))

    def _weight_update(self):
        if self.loss == 'ISE':
            # quadratic programming for updating w
            diff = self.reduced_means[
                np.newaxis, :] - self.reduced_means[:, np.newaxis]
            covs = self.reduced_covs[
                np.newaxis, :] + self.reduced_covs[:, np.newaxis]
            self.H1 = np.exp(log_normal(diff, covs))

            diff = self.reduced_means[np.newaxis, :] - self.means[:,
                                                                  np.newaxis]
            covs = self.reduced_covs[np.newaxis, :] + self.covs[:, np.newaxis]
            self.H2 = np.exp(log_normal(diff, covs))

            P = matrix(self.H1, tc='d')
            q = matrix(-self.weights.T.dot(self.H2), tc='d')
            G = matrix(-np.eye(self.new_n), tc='d')
            h = matrix(np.zeros((self.new_n)), tc='d')
            A = matrix(np.ones((1, self.new_n)), tc='d')
            b = matrix(np.array([1]), tc='d')
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)
            self.reduced_weights = np.array(sol['x']).reshape((-1, ))
            return 2 * sol['primal objective']

        else:
            # use BFGS method to update the weights
            obj_lambda = lambda softmaxw: opt_obj(self.reduced_means,
                                                  self.reduced_covs,
                                                  softmax(softmaxw),
                                                  self.means,
                                                  self.covs,
                                                  self.weights,
                                                  chol=False,
                                                  loss=self.loss)

            grad_lambda = lambda softmaxw: obj_grads_w(
                self.reduced_means, self.reduced_covs, softmax(
                    softmaxw), self.means, self.covs, self.weights, self.loss)

            res = optimize.minimize(obj_lambda,
                                    np.log(self.reduced_weights),
                                    method='BFGS',
                                    jac=grad_lambda,
                                    options={
                                        'gtol': 1e-5,
                                        'disp': True
                                    })
            # print(res)
            self.reduced_weights = softmax(res.x)
            return res.fun

        # return self._obj()

    def _support_update(self):
        # update the mean and covariance with gradient descent
        # for the covariance, optimize its cholesky decomposition
        # to ensure positive definiteness
        n, d = self.new_n, self.d
        obj_lambda = lambda x: opt_obj(x[:n * d].reshape(
            (n, d)), x[n * d:].reshape(
                (n, d, d)), self.reduced_weights, self.means, self.covs, self.
                                       weights, self.loss)

        grad_lambda = lambda x: obj_grads_theta(x[:n * d].reshape((n, d)),
                                                x[n * d:].reshape((n, d, d)),
                                                self.reduced_weights,
                                                self.means,
                                                self.covs,
                                                self.weights,
                                                loss=self.loss)
        self.reduced_covs_chol = np.zeros_like(self.reduced_covs)
        for i, cov in enumerate(self.reduced_covs):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                raise ValueError("covariance chol is wrong.")
            self.reduced_covs_chol[i] = cov_chol

        x0 = np.concatenate((self.reduced_means.reshape(
            (-1, )), self.reduced_covs_chol.reshape((-1, ))))
        res = optimize.minimize(obj_lambda,
                                x0,
                                method='BFGS',
                                jac=grad_lambda,
                                options={
                                    'gtol': 1e-5,
                                    'disp': True
                                })
        # print(res)

        self.reduced_means, self.reduced_covs_chol = res.x[:n * d].reshape(
            (n, d)), res.x[n * d:].reshape((n, d, d))
        for i in range(self.new_n):
            self.reduced_covs[i] = self.reduced_covs_chol[i].dot(
                self.reduced_covs_chol[i].T)
        return res.fun
        # self._update_H1_H2()
        # return self._obj()

    # def _update_H1_H2(self):
    #     diff = self.reduced_means[
    #         np.newaxis, :] - self.reduced_means[:, np.newaxis]
    #     covs = self.reduced_covs[np.newaxis, :] + self.reduced_covs[:,
    #                                                                 np.newaxis]
    #     self.H1 = np.exp(log_normal(diff, covs))
    #     diff = self.reduced_means[np.newaxis, :] - self.means[:, np.newaxis]
    #     covs = self.reduced_covs[np.newaxis, :] + self.covs[:, np.newaxis]
    #     self.H2 = np.exp(log_normal(diff, covs))

    #     if self.loss == 'NISE':
    #         diff = self.means[np.newaxis, :] - self.means[:, np.newaxis]
    #         covs = self.covs[np.newaxis, :] + self.covs[:, np.newaxis]
    #         self.H3 = np.exp(log_normal(diff, covs))

    def iterative(self):
        self._initialize_parameter()
        # print(self.reduced_weights)
        obj = np.Inf
        proc_time = time.time()
        for n_iter in range(1, self.max_iter + 1):
            print('Iteration %d' % n_iter)
            current_time = time.time()
            obj_current = self._weight_update()
            # print(obj_current)
            if min(self.reduced_weights) == 0:
                warnings.warn('The reduced mixture has fewer components!')
            else:
                change = obj - obj_current
                # print(obj, obj_current, change)
                print(time.time() - current_time)
                current_time = time.time()
                if abs(change) < self.tol:
                    self.converged_ = True
                    self.obj = obj
                    self.n_iter_ = n_iter
                    break
                if change < 0.0:
                    raise ValueError('The objective function is increasing!')
                obj = obj_current
                obj_current = self._support_update()
                change = obj - obj_current
                # print(obj, obj_current, change)
                print(time.time() - current_time)
                if abs(change) < self.tol:
                    self.converged_ = True
                    self.obj = obj
                    self.n_iter_ = n_iter
                    break
                if change < 0.0:
                    raise ValueError('The objective function is increasing!')
                obj = obj_current

        self.time_ = time.time() - proc_time
        print(self.time_, self.n_iter_)
        if not self.converged_:
            warnings.warn('Did not converge. Try different init parameters, '
                          'or increase max_iter, tol ')


class GMR_opt_BFGS:
    """
    Find a GMM with n_components that is closest
    to a GMM parameterized by means, covs, weights 
    by optimization based method.
    The distances implemented are ISE, NISE, and CS

    Parameters
    ----------
    means : numpy array, (N, d) 
    covs :  numpy array, (N, d, d) 
    weights: numpy array, (N, )
    Returns
    -------
    weights and support points of reduced GMM.
    """
    def __init__(self,
                 means,
                 covs,
                 weights,
                 n,
                 loss='ISE',
                 init_method='Runnalls',
                 tol=1e-8,
                 max_iter=100,
                 random_state=0,
                 means_init=None,
                 covs_init=None,
                 weights_init=None):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.tol = tol
        self.n = self.weights.shape[0]
        self.m = n
        self.d = means.shape[1]
        self.random_state = random_state
        self.init_method = init_method
        self.loss = loss
        self.reduced_means = means_init
        self.reduced_covs = covs_init
        self.reduced_weights = weights_init

    def _initialize_parameter(self):
        """Initializatin of the reduced mixture"""
        # self.H1 = np.zeros((self.new_n, self.new_n))
        # self.H2 = np.zeros((self.origin_n, self.new_n))
        # if self.loss == 'NISE':
        #     self.H3 = np.zeros((self.origin_n, self.origin_n))

        if self.init_method == 'kmeans':
            total_sample_size = 10000
            X = rmixGaussian(self.means, self.covs, self.weights,
                             total_sample_size, self.random_state)[0]
            gm = GaussianMixture(n_components=self.m, random_state = self.random_state, tol = 1e-6).fit(X)
            self.reduced_means = gm.means_
            self.reduced_covs = gm.covariances_
            self.reduced_weights = gm.weights_
        elif self.init_method == 'user':
            pass
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.m, self.init_method)

    def run(self):
        self._initialize_parameter()
        proc_time = time.time()
        obj_lambda = lambda x: opt_obj(x[self.m:
                                         (self.m + self.m * self.d)].reshape(
                                             (self.m, self.d)),
                                       x[(self.m + self.m * self.d):].reshape(
                                           (self.m, self.d, self.d)),
                                       softmax(x[:self.m]),
                                       self.means,
                                       self.covs,
                                       self.weights,
                                       loss=self.loss)

        grad_lambda = lambda x: obj_grads(
            x[self.m:(self.m + self.m * self.d)].reshape((self.m, self.d)),
            x[(self.m + self.m * self.d):].reshape((self.m, self.d, self.d)),
            softmax(x[:self.m]),
            self.means,
            self.covs,
            self.weights,
            loss=self.loss)

        self.reduced_covs_chol = np.zeros_like(self.reduced_covs)
        for i, cov in enumerate(self.reduced_covs):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                raise ValueError('covariance chol is wrong.')
            self.reduced_covs_chol[i] = cov_chol

        x0 = np.concatenate(
            (np.log(self.reduced_weights), self.reduced_means.reshape(
                (-1, )), self.reduced_covs_chol.reshape((-1, ))))
        res = optimize.minimize(obj_lambda,
                                x0,
                                method='BFGS',
                                jac=grad_lambda,
                                options={'gtol': self.tol})
        if res.success:
            self.converged_ = True
            self.obj = res.fun
            self.reduced_weights = softmax(res.x[:self.m])
            self.reduced_means = res.x[self.m:(self.m + self.m * self.d)].reshape(
                (self.m, self.d))
            self.reduced_covs = res.x[(self.m + self.m * self.d):].reshape(
                (self.m, self.d, self.d))
            for i, cov in enumerate(self.reduced_covs):
                self.reduced_covs[i] = cov.dot(cov.T)
        else:
            self.converged_ = False
            print(res.message)
            self.res = res
            self.obj = res.fun
            self.reduced_weights = softmax(res.x[:self.m])
            self.reduced_means = res.x[self.m:(self.m + self.m * self.d)].reshape(
                (self.m, self.d))
            self.reduced_covs = res.x[(self.m + self.m * self.d):].reshape(
                (self.m, self.d, self.d))
            for i, cov in enumerate(self.reduced_covs):
                self.reduced_covs[i] = cov.dot(cov.T)
                
        self.time_ = time.time() - proc_time
        self.n_iter_ = res.nit

        # print(self.time_)
        
        if not self.converged_:
            warnings.warn('Did not converge. Try different init parameters, '
                          'or increase max_iter, tol ')


if __name__ == '__main__':
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    # means = np.array(
    #     [1.45, 2.2, 0.67, 0.48, 1.49, 0.91, 1.01, 1.42, 2.77, 0.89]).reshape(
    #         (-1, 1))
    # covs = np.array([
    #     0.0487, 0.0305, 0.1171, 0.0174, 0.0295, 0.0102, 0.0323, 0.0380, 0.0115,
    #     0.0679
    # ]).reshape((-1, 1, 1))
    # weights = np.array(
    #     [0.03, 0.18, 0.12, 0.19, 0.02, 0.16, 0.06, 0.1, 0.08, 0.06])
    # M = 5
    means = np.array([-1.0, 2]).reshape((-1, 1))
    covs = np.array([.15, .15]).reshape((-1, 1, 1))
    weights = np.array([.45, .5])
    M = 1

    # reduction = GMR_CTD(means,
    #              covs,
    #              weights,
    #              5,
    #              init_method="kmeans",
    #              tol=1e-5,
    #              max_iter=100,
    #              ground_distance="KL",
    #              reg=0,
    #              means_init=None,
    #              covs_init=None,
    #              weights_init=None,
    #              random_state=0,
    #              coeff=None)
    reduction = GMR_L2(means,
                       covs,
                       weights,
                       M,
                       False,
                       init_method="kmeans",
                       tol=1e-5,
                       max_iter=100)

    reduction.iterative()
    # print(
    #     GMM_L2([means, reduction.reduced_means],
    #            [covs, reduction.reduced_covs],
    #            [weights, reduction.reduced_weights]))

    # visualization
    reduced_means = np.squeeze(reduction.reduced_means)
    reduced_covs = np.squeeze(reduction.reduced_covs)
    # idx = np.argsort(reduced_means)
    # reduced_means = reduced_means[idx]
    # reduced_covs = reduced_covs[idx]
    # reduced_weights = reduction.reduced_weights
    # reduced_weights = reduced_weights[idx]

    # print(means)
    # print(reduced_means)

    # print(covs)
    # print(reduced_covs)

    # print(weights)
    # print(reduced_weights)

    x = np.linspace(-10, 10, 100)
    y2 = dmixf(x, reduced_means, np.sqrt(reduced_covs),
               reduction.reduced_weights, norm)

    reduction = GMR_L2(means,
                       covs,
                       weights,
                       M,
                       True,
                       init_method="kmeans",
                       tol=1e-5,
                       max_iter=100)

    reduction.iterative()

    print(
        GMM_L2([means, reduction.reduced_means],
               [covs, reduction.reduced_covs],
               [weights, reduction.reduced_weights]))

    # visualization
    reduced_means = np.squeeze(reduction.reduced_means)
    reduced_covs = np.squeeze(reduction.reduced_covs)
    reduced_weights = reduction.reduced_weights
    y3 = dmixf(x, reduced_means, np.sqrt(reduced_covs), reduced_weights, norm)

    means = np.squeeze(means)
    covs = np.squeeze(covs)
    y1 = dmixf(x, means, np.sqrt(covs), weights, norm)
    # idx = np.argsort(means)
    # means = means[idx]
    # covs = covs[idx]
    # weights = weights[idx]

    plt.figure()
    plt.plot(x, y1, label='original')
    plt.plot(x, y2, label='ISE')
    plt.plot(x, y3, label='NISE')

    plt.legend()
    plt.savefig('ISE_vs_NISE.png')
