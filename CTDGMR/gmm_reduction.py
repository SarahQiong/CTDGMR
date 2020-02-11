import ot
import time
import warnings
import numpy as np
from scipy import linalg
from scipy import optimize
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from numpy.linalg import det
from cvxopt import matrix
from cvxopt import solvers
from sklearn.cluster import KMeans
from sklearn.mixture.base import _check_shape
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from .utils import GMM_L2, GMM_CWD, Elogp, MC_ll
from .utils import GMM_sampler, Gaussian_barycenter, Gaussian_distance
from .utils import moment_preserving_merge, wbarycenter_merge, bound_on_KL
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


"""
Minimum L2 distance estimator
"""


def ISE(reduced_means, reduced_covs_chol, reduced_weights, means, covs,
        weights):
    k2 = reduced_means.shape[0]
    reduced_covs = np.zeros_like(reduced_covs_chol)
    for i in range(k2):
        reduced_covs[i] = reduced_covs_chol[i].dot(reduced_covs_chol[i].T)
    return GMM_L2([means, reduced_means], [covs, reduced_covs],
                  [weights, reduced_weights])


class GMR_L2:
    """Find a GMM with n_components that is closest
    to a GMM parameterized by means, covs, weights in
    the L2 distance.

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
                 init_method="Runnalls",
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

    def _initialize_parameter(self):
        """Initializatin of the Gaussian mixture barycenter"""
        self.H1 = np.zeros((self.new_n, self.new_n))
        self.H2 = np.zeros((self.origin_n, self.new_n))
        self.H3 = np.zeros((self.origin_n, self.origin_n))

        if self.init_method == "kmeans":
            total_sample_size = 10000
            X = GMM_sampler(self.means, self.covs, self.weights,
                            total_sample_size, self.random_state)[0]
            kmeans = KMeans(n_clusters=self.new_n,
                            n_init=1,
                            random_state=self.random_state).fit(X)
            self.reduced_means = kmeans.cluster_centers_
            self.reduced_covs = np.tile(np.mean(self.covs, 0),
                                        (self.new_n, 1, 1))
        elif self.init_method == "user":
            self.reduced_means, self.reduced_covs, self.reduced_weights = reduced_means, reduced_covs, reduced_weights
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.new_n,
                self.init_method)
        self._update_H1_H2()

    def _obj(self):
        return np.dot(self.reduced_weights.T, self.H1).dot(self.reduced_weights) - \
               2*np.dot(self.weights.T, self.H2).dot(self.reduced_weights) + \
               np.dot(self.weights.T, self.H3).dot(self.weights)

    def _weight_update(self):
        # quadratic programming for updating w
        P = matrix(self.H1, tc='d')
        q = matrix(-self.weights.T.dot(self.H2), tc='d')
        G = matrix(-np.eye(self.new_n), tc='d')
        h = matrix(np.zeros((self.new_n)), tc='d')
        A = matrix(np.ones((1, self.new_n)), tc='d')
        b = matrix(np.array([1]), tc='d')
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        self.reduced_weights = np.array(sol['x']).reshape((-1, ))
        return self._obj()

    def _support_update(self):
        # update the mean and covariance with gradient descent
        n, d = self.new_n, self.d
        ISE_lambda = lambda x: ISE(x[:n * d].reshape(
            (n, d)), x[n * d:].reshape((n, d, d)), self.reduced_weights, self.
                                   means, self.covs, self.weights)
        x0 = np.concatenate((self.reduced_means.reshape(
            (-1, )), np.sqrt(self.reduced_covs).reshape((-1, ))))
        res = optimize.minimize(ISE_lambda, x0, method='BFGS')
        self.reduced_means, self.reduced_covs_chol = res.x[:n * d].reshape(
            (n, d)), res.x[n * d:].reshape((n, d, d))
        for i in range(self.new_n):
            self.reduced_covs[i] = self.reduced_covs_chol[i].dot(
                self.reduced_covs_chol[i].T)
        self._update_H1_H2()
        return self._obj()

    def _update_H1_H2(self):
        for i in range(self.H1.shape[0]):
            for j in range(self.H1.shape[1]):
                self.H1[i, j] = multivariate_normal.pdf(
                    self.reduced_means[i],
                    mean=self.reduced_means[j],
                    cov=self.reduced_covs[i] + self.reduced_covs[j])
        for i in range(self.H2.shape[0]):
            for j in range(self.H2.shape[1]):
                self.H2[i, j] = multivariate_normal.pdf(
                    self.means[i],
                    mean=self.reduced_means[j],
                    cov=self.covs[i] + self.reduced_covs[j])
        for i in range(self.H3.shape[0]):
            for j in range(self.H3.shape[1]):
                self.H3[i, j] = multivariate_normal.pdf(self.means[i],
                                                        mean=self.means[j],
                                                        cov=self.covs[i] +
                                                        self.covs[j])

    def iterative(self):
        self._initialize_parameter()
        obj = np.Inf
        for n_iter in range(1, self.max_iter + 1):
            prev_obj = obj
            obj1 = self._weight_update()
            if min(self.reduced_weights) == 0:
                self.new_n -= 1
                self.iterative()
            else:
                obj2 = self._support_update()
                if obj2 > obj1:
                    raise ValueError(
                        "Warning: The objective function is increasing!")
                obj = obj2
                change = obj - prev_obj
                if abs(change) < self.tol:
                    self.converged_ = True
                    self.obj = obj
                    self.n_iter_ = n_iter
                    break
        if not self.converged_:
            warnings.warn(
                'Did not converge. Try different init parameters, '
                'or increase max_iter, tol ', ConvergenceWarning)


"""
Clustering based algorithm
"""


class GMR_clustering:
    """Clustering based algorithm for GMR

    Parameters
    ----------
    means : numpy array, (N, d) 
    covs :  numpy array, (N, d, d) 
    weights: numpy array, (N, )
    method: string
    dphem
    hem

    Returns
    -------
    weights and support points of reduced GMM.
    """
    def __init__(self,
                 means,
                 covs,
                 weights,
                 n,
                 n_pseudo=1,
                 method="dphem",
                 init_method="kmeans",
                 tol=1e-5,
                 max_iter=100,
                 means_init=None,
                 covs_init=None,
                 weights_init=None,
                 random_state=0):

        self.means = means
        self.covs = covs
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = n
        self.Nvs = n_pseudo
        self.method = method
        self.random_state = random_state
        self.converged_ = False
        self.init_method = init_method
        self.means_init = means_init
        self.covs_init = covs_init
        self.weights_init = weights_init

    def _initialize_cluster_center(self):
        """Initializatin of the Gaussian mixture barycenter"""
        #kmeans initializaion of Barycenter mean
        if self.init_method == "kmeans":
            total_sample_size = 10000
            X = GMM_sampler(self.means, self.covs, self.weights,
                            total_sample_size, self.random_state)[0]
            kmeans = KMeans(n_clusters=self.new_n,
                            n_init=1,
                            random_state=self.random_state).fit(X)
            self.reduced_means = kmeans.cluster_centers_
            self.reduced_covs = np.tile(np.mean(self.covs, 0),
                                        (self.new_n, 1, 1))
            self.reduced_weights = np.array([
                np.sum(kmeans.labels_ == i) / total_sample_size
                for i in range(self.new_n)
            ])
        elif self.init_method == "user":
            self.reduced_means = self.means_init
            self.reduced_covs = self.covs_init
            self.reduced_weights = self.weights_init
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.new_n,
                self.init_method)

    def _obj(self):
        if self.method == "dphem":
            temp = self.zij * (np.log(self.reduced_weights) -
                               np.log(self.zij) + self.Nvs * self.Elogp)
            # temp1 = self.zij * (np.log(self.reduced_weights) -
            #                    np.log(self.zij))
            # print(np.sum(temp1.T.dot(self.weights)))


            means = [self.means, self.reduced_means]
            covs = [self.covs, self.reduced_covs]
            weights = [self.weights, self.reduced_weights]
            KL = MC_ll(means, covs, weights)
            a = self.Nvs*KL
            b = np.sum(temp.T.dot(self.weights))
            temp1 = self.zij * (np.log(self.reduced_weights) -
                               np.log(self.zij) + self.Elogp)
            c = self.Nvs * (np.sum(temp1.T.dot(self.weights)))
            # print(a,b, c, a>b)

            return np.sum(temp.T.dot(self.weights))

        elif self.method == "hem":
            M = self.Nvs * self.weights
            return np.sum(self.Elogp.T.dot(M))

        else:
            raise ValueError("The %s reduction method is not implement!" %
                             self.method)

    def _E_step(self):
        # compute zij
        # compute E_{X\sim \phi(x;H^i)}(log\phi(X;G^j))
        self.Elogp = Elogp(self.means, self.covs, self.reduced_means,
                           self.reduced_covs)

        if self.method == 'hem':
            lognum = ((self.Nvs * self.Elogp).T * self.weights).T + np.log(
                self.reduced_weights)
            logzij = (lognum.T - logsumexp(lognum, axis=1)).T
            self.zij = np.exp(logzij)

        elif self.method == 'dphem':
            lognum = self.Nvs * self.Elogp + np.log(self.reduced_weights)
            logzij = (lognum.T - logsumexp(lognum, axis=1)).T
            self.logzij = logzij
            self.zij = np.exp(logzij)

        else:
            raise ValueError("The %s reduction method is not implement!" %
                             self.method)
        return self._obj()

    def _M_step(self):
        if self.method == "hem":
            self.reduced_weights = np.sum(self.zij, axis=0) / self.origin_n
            normalization = np.sum((self.zij.T * self.weights).T, axis=0)
            resp = (self.zij.T * self.weights).T
            self.reduced_means = resp.T.dot(
                self.means) / normalization[:, np.newaxis]
            for i in range(self.new_n):
                diff = self.means - self.reduced_means[i]
                self.reduced_covs[i] = np.dot(resp[:, i] * diff.T, diff)
                self.reduced_covs[i] += (self.covs.T.dot(resp[:, i])).T
                self.reduced_covs[i] /= normalization[i]

        elif self.method == "dphem":
            self.reduced_weights = np.sum((self.zij.T * self.weights).T,
                                          axis=0)
            resp = (self.zij.T * self.weights).T
            self.reduced_means = resp.T.dot(
                self.means) / self.reduced_weights[:, np.newaxis]
            for i in range(self.new_n):
                diff = self.means - self.reduced_means[i]
                self.reduced_covs[i] = np.dot(resp[:, i] * diff.T, diff)
                self.reduced_covs[i] += (self.covs.T.dot(resp[:, i])).T
                self.reduced_covs[i] /= self.reduced_weights[i]
            # print("M step")
            # print(self.reduced_means, self.reduced_covs, self.reduced_weights)

        else:
            raise ValueError("The %s reduction method is not implement!" %
                             self.method)
        # update Eij matrix to evaluate the objective function
        self.Elogp = Elogp(self.means, self.covs, self.reduced_means,
                           self.reduced_covs)

        return self._obj()

    def iterative(self):
        self._initialize_cluster_center()
        obj = np.Inf
        for n_iter in range(1, self.max_iter + 1):
            # print("Iteration %d" % n_iter)
            # print(self.reduced_means, self.reduced_covs, self.reduced_weights)
            prev_obj = obj
            obj1 = self._E_step()
            obj2 = self._M_step()
            if obj2 < obj1:
                raise ValueError(
                    "Warning: The objective function is decreasing!")
            obj = obj2
            change = obj - prev_obj
            if abs(change) < self.tol:
                self.converged_ = True
                self.obj = obj
                self.n_iter_ = n_iter
                break
        if not self.converged_:
            warnings.warn(
                'EM algorithm did not converge. '
                'Try different init parameters, '
                'or increase max_iter, tol ', ConvergenceWarning)


"""
Minimum composite Wasserstein distance for GMR

"""


def entropy(log_ot_plan):
    """
    The entropy of a coupling matrix
    """

    return 1 - np.sum(np.exp(log_ot_plan) * log_ot_plan)


class GMR_CTD:
    """Find a GMM with n_components that is closest
    to a GMM parameterized by means, covs, weights in
    the composite transportation distance sense.

    Parameters
    ----------
    reg: strength of entropic regularization

    Returns
    -------
    weights and support points of reduced GMM.
    """
    def __init__(self,
                 means,
                 covs,
                 weights,
                 n,
                 n_pseudo=1,
                 init_method="kmeans",
                 tol=1e-5,
                 max_iter=100,
                 ground_distance="W2",
                 reg=0,
                 means_init=None,
                 covs_init=None,
                 weights_init=None,
                 random_state=0):

        self.means = means
        self.covs = covs
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = n
        self.n_pseudo = n_pseudo
        self.random_state = random_state
        self.ground_distance = ground_distance
        self.converged_ = False
        if reg >= 0:
            self.reg = reg
        else:
            raise ValueError("The regularization term should be non-negative.")
        self.init_method = init_method
        self.means_init = means_init
        self.covs_init = covs_init
        self.weights_init = weights_init

    def _initialize_parameter(self):
        """Initializatin of the clustering barycenter"""
        if self.init_method == "kmeans":
            total_sample_size = 10000
            X = GMM_sampler(self.means, self.covs, self.weights,
                            total_sample_size, self.random_state)[0]
            kmeans = KMeans(n_clusters=self.new_n,
                            n_init=1,
                            random_state=self.random_state).fit(X)
            self.reduced_means = kmeans.cluster_centers_
            self.reduced_covs = np.tile(np.mean(self.covs, 0),
                                        (self.new_n, 1, 1))
            self.reduced_weights = np.array([
                np.sum(kmeans.labels_ == i) / total_sample_size
                for i in range(self.new_n)
            ])
        elif self.init_method == "user":
            self.reduced_means = self.means_init
            self.reduced_covs = self.covs_init
            self.reduced_weights = self.weights_init
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.new_n,
                self.init_method)
        self.cost_matrix = GMM_CWD([self.means, self.reduced_means],
                                   [self.covs, self.reduced_covs],
                                   [self.weights, self.reduced_weights],
                                   ground_distance=self.ground_distance,
                                   matrix=True,
                                   N=self.n_pseudo)
        # print(self.cost_matrix.shape)

    def _obj(self):
        if self.reg == 0:
            return np.sum(self.cost_matrix * self.ot_plan)
        elif self.reg > 0:
            return np.sum(self.cost_matrix *
                          self.ot_plan) - self.reg * entropy(self.log_ot_plan)

    def _weight_update(self):
        if self.reg == 0:
            self.clustering_matrix = (self.cost_matrix.T == np.min(
                self.cost_matrix, 1)).T
            self.ot_plan = self.clustering_matrix * self.weights.reshape(
                (-1, 1))
            self.reduced_weights = self.ot_plan.sum(axis=0)
        elif self.reg > 0:
            lognum = -self.cost_matrix / self.reg
            logtemp = (lognum.T - logsumexp(lognum, axis=1)).T
            self.log_ot_plan = (logtemp.T + np.log(self.weights)).T
            self.ot_plan = np.exp(self.log_ot_plan)
            # self.ot_plan = (np.exp(logtemp).T * self.weights).T
            self.reduced_weights = self.ot_plan.sum(axis=0)
        return self._obj()

    def _support_update(self):
        for i in range(self.new_n):
            self.reduced_means[i], self.reduced_covs[i] = Gaussian_barycenter(
                self.means,
                self.covs,
                self.ot_plan[:, i],
                ground_distance=self.ground_distance)
        self.cost_matrix = GMM_CWD([self.means, self.reduced_means],
                                   [self.covs, self.reduced_covs],
                                   [self.weights, self.reduced_weights],
                                   ground_distance=self.ground_distance,
                                   matrix=True,
                                   N=self.n_pseudo)
        return self._obj()

    def iterative(self):
        self._initialize_parameter()
        obj = np.Inf
        proc_time = time.time()
        for n_iter in range(1, self.max_iter + 1):
            # print("Iteration %d" % n_iter)
            # print(self.reduced_means, self.reduced_covs, self.reduced_weights)
            prev_obj = obj
            obj1 = self._weight_update()
            # print("weight update", obj1)
            # print("assignment matrix", self.ot_plan)
            if min(self.ot_plan.sum(axis=0)) == 0:
                self.new_n -= 1
                self.iterative()
            else:
                obj2 = self._support_update()
                # print("support update", obj2)
                change = obj2 - obj1
                obj = obj2
                if abs(change) < self.tol:
                    self.converged_ = True
                    self.obj = obj
                    self.n_iter_ = n_iter
                    break
                if obj2 > obj1:
                    raise ValueError(
                        "Warning: The objective function is increasing!")
        self.time = time.time() - proc_time
        if not self.converged_:
            print('Algorithm did not converge. '
                  'Try different init parameters, '
                  'or increase max_iter, tol ')
