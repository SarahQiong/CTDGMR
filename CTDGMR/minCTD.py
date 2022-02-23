import ot
import time
import warnings
import numpy as np
from scipy import linalg
from scipy import optimize
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from .distance import GMM_L2, GMM_CTD, Gaussian_distance
from .greedy import *
from .utils import *
from .barycenter import barycenter
"""
Minimum composite transportation divergence (CTD) for GMR

Created by Qiong Zhang
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
                 n_pseudo=100,
                 init_method='kmeans',
                 tol=1e-5,
                 max_iter=100,
                 ground_distance='W2',
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
            raise ValueError('The regularization term should be non-negative.')
        self.init_method = init_method
        self.means_init = means_init
        self.covs_init = covs_init
        self.weights_init = weights_init
        self.time_ = []

    def _initialize_parameter(self):
        """Initializatin of the clustering barycenter"""
        if self.init_method == 'kmeans':
            total_sample_size = 10000
            X = rmixGaussian(self.means, self.covs, self.weights, total_sample_size,
                             self.random_state)[0]
            gm = GaussianMixture(n_components=self.new_n, random_state=self.random_state,
                                 tol=1e-6).fit(X)
            self.reduced_means = gm.means_
            self.reduced_covs = gm.covariances_
            self.reduced_weights = gm.weights_

        elif self.init_method == 'user':
            self.reduced_means = self.means_init
            self.reduced_covs = self.covs_init
            self.reduced_weights = self.weights_init
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.new_n, self.init_method)
        
        self.cost_matrix = GMM_CTD(means=[self.means, self.reduced_means],
                                   covs=[self.covs, self.reduced_covs],
                                   weights=[self.weights, self.reduced_weights],
                                   ground_distance=self.ground_distance,
                                   matrix=True,
                                   N=self.n_pseudo)

    def _obj(self):
        if self.reg == 0:
            return np.sum(self.cost_matrix * self.ot_plan)
        elif self.reg > 0:
            return np.sum(self.cost_matrix * self.ot_plan) - self.reg * entropy(self.log_ot_plan)

    def _weight_update(self):
        if self.reg == 0:
            self.clustering_matrix = (self.cost_matrix.T == np.min(self.cost_matrix, 1)).T
            self.ot_plan = self.clustering_matrix * (self.weights /
                                                     self.clustering_matrix.sum(1)).reshape((-1, 1))
            # if there are ties, then the weights are equally splitted into
            # different groups
            self.reduced_weights = self.ot_plan.sum(axis=0)
        elif self.reg > 0:
            lognum = -self.cost_matrix / self.reg
            logtemp = (lognum.T - logsumexp(lognum, axis=1)).T
            self.log_ot_plan = (logtemp.T + np.log(self.weights)).T
            self.ot_plan = np.exp(self.log_ot_plan)
            self.reduced_weights = self.ot_plan.sum(axis=0)
        return self._obj()

    def _support_update(self):
        for i in range(self.new_n):
            self.reduced_means[i], self.reduced_covs[i] = barycenter(
                self.means,
                self.covs,
                self.ot_plan[:, i],
                mean_init=self.reduced_means[i],
                cov_init=self.reduced_covs[i],
                ground_distance=self.ground_distance)
        self.cost_matrix = GMM_CTD([self.means, self.reduced_means], [self.covs, self.reduced_covs],
                                   [self.weights, self.reduced_weights],
                                   ground_distance=self.ground_distance,
                                   matrix=True,
                                   N=self.n_pseudo)
        return self._obj()

    def iterative(self):
        self._initialize_parameter()
        obj = np.Inf
        for n_iter in range(1, self.max_iter + 1):
            proc_time = time.time()
            obj_current = self._weight_update()
            # remove the empty cluster centers
            index = np.where(self.ot_plan.sum(axis=0) != 0)
            self.new_n = index[0].shape[0]
            self.ot_plan = self.ot_plan.T[index].T
            self.reduced_means = self.reduced_means[index[0]]
            self.reduced_covs = self.reduced_covs[index[0]]
            self.reduced_weights = self.reduced_weights[index[0]]
            change = obj - obj_current
            if abs(change) < self.tol:
                self.converged_ = True
                self.obj = obj
                self.n_iter_ = n_iter
                break
            if change < 0.0:
                raise ValueError('Weight update: The objective function is increasing!')
            obj = obj_current
            obj_current = self._support_update()
            change = obj - obj_current
            self.time_.append(time.time() - proc_time)
            if abs(change) < self.tol:
                self.converged_ = True
                self.obj = obj
                self.n_iter_ = n_iter
                break
            if change < 0.0:
                raise ValueError('Support update: The objective function is increasing!')
            obj = obj_current

        if not self.converged_:
            print('Algorithm did not converge. '
                  'Try different init parameters, '
                  'or increase max_iter, tol ')


if __name__ == '__main__':
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    means = np.array([1.45, 2.2, 0.67, 0.48, 1.49, 0.91, 1.01, 1.42, 2.77, 0.89]).reshape((-1, 1))
    covs = np.array(
        [0.0487, 0.0305, 0.1171, 0.0174, 0.0295, 0.0102, 0.0323, 0.0380, 0.0115, 0.0679]).reshape(
            (-1, 1, 1))
    weights = np.array([0.03, 0.18, 0.12, 0.19, 0.02, 0.16, 0.06, 0.1, 0.08, 0.06])

    print(
        GMM_L2([
            means,
            np.array([0.48576481, 0.91249295, 1.03276885, 1.39806918, 2.20693554, 2.76902991
                      ]).reshape((-1, 1))
        ], [
            covs,
            np.array([0.01860878, 0.01370735, 0.29000884, 0.03605234, 0.02781392, 0.0116604
                      ]).reshape((-1, 1, 1))
        ], [
            weights,
            np.array([0.22000596, 0.23553842, 0.18454371, 0.11243351, 0.16731566, 0.08016274])
        ]))


    reduction = GMR_CTD(
        means,
        covs,
        weights,
        5,
        init_method="user",
        tol=1e-5,
        max_iter=100,
        ground_distance="L2",
        reg=0,
        means_init=np.array(
            [0.48576481, 0.91249295, 1.03276885, 1.39806918, 2.20693554, 2.76902991]).reshape(
                (-1, 1)),
        covs_init=np.array([0.01860878, 0.01370735, 0.29000884, 0.03605234, 0.02781392,
                            0.0116604]).reshape((-1, 1, 1)),
        weights_init=np.array(
            [0.22000596, 0.23553842, 0.18454371, 0.11243351, 0.16731566, 0.08016274]),
        random_state=0,
        coeff=None)
    reduction.iterative()

  
    print(
        GMM_L2([means, reduction.reduced_means], [covs, reduction.reduced_covs],
               [weights, reduction.reduced_weights]))

    reduction2 = GMR_CTD(
        means,
        covs,
        weights,
        5,
        init_method="user",
        tol=1e-5,
        max_iter=100,
        ground_distance="SW",
        reg=0,
        means_init=np.array(
            [0.48576481, 0.91249295, 1.03276885, 1.39806918, 2.20693554, 2.76902991]).reshape(
                (-1, 1)),
        covs_init=np.array([0.01860878, 0.01370735, 0.29000884, 0.03605234, 0.02781392,
                            0.0116604]).reshape((-1, 1, 1)),
        weights_init=np.array(
            [0.22000596, 0.23553842, 0.18454371, 0.11243351, 0.16731566, 0.08016274]),
        random_state=0,
        coeff=None)
    reduction2.iterative()



    print(
        GMM_L2([means, reduction2.reduced_means], [covs, reduction2.reduced_covs],
               [weights, reduction2.reduced_weights]))

    # visualization
    means = np.squeeze(means)
    covs = np.squeeze(covs)
    reduced_means = np.squeeze(reduction.reduced_means)
    reduced_covs = np.squeeze(reduction.reduced_covs)

    reduced_means2 = np.squeeze(reduction2.reduced_means)
    reduced_covs2 = np.squeeze(reduction2.reduced_covs)

    x = np.linspace(0, 3, 100)

    y1 = dmixf(x, means, np.sqrt(covs), weights, norm)
    y2 = dmixf(x, reduced_means, np.sqrt(reduced_covs), reduction.reduced_weights, norm)
    y3 = dmixf(x, reduced_means2, np.sqrt(reduced_covs2), reduction2.reduced_weights, norm)

    plt.figure()
    plt.plot(x, y1, label='original')
    plt.plot(x, y2, label='reduced (L2)')
    plt.plot(x, y3, label='reduced (SW)')
    plt.legend()
    plt.savefig('test.png')
