import numpy as np
import argparse
from scipy import stats as ss
from CTDGMR.gmm_reduction import *
from CTDGMR.utils import *
import _pickle as pickle
import time


def gaussian_product(means, covs):
    """
    Product of two Gaussians is a scaled 
    Gaussian. This function gives the mean 
    and cov of the scaled Gaussian.

    Normalization constant is ignored

    Input: 
    means: array like (n,d)
    covs: array like (n, d, d)

    Output:
    mu: array like (d,)
    sigma: array like (d,d)

    """
    # If each cov is diagonal

    n, d = means.shape
    if d == 1:
        means = means.reshape((-1, ))
        covs = covs.reshape((-1))
        sigma = 1 / np.sum(1 / covs)
        mu = np.sum(means / covs) / sigma
        sigma = np.array(sigma).reshape((1, 1))
        mu = np.array(mu).reshape((1, ))
    else:
        pres = 1 / np.einsum('kii->ki', covs)
        if np.array_equal(np.sum(pres, 0), np.sum(abs(covs), (1, 2))):
            mu = np.sum(pres * means, 0)
            mu = sigma * mu
            sigma = np.diag(1 / np.sum(pres, 0))
        else:
            pres = np.zeros_like(covs)
            mu = np.array(means.shape[0])
            for n in range(covs.shape[0]):
                pres[n] = np.linalg.inv(covs[n])
                mu += pres[n] @ means[n]
            sigma = np.linalg.inv(np.sum(pres, 0))
            mu = sigma @ mu
    return (mu, sigma)


def GMM_2prod(means, covs, weights, normalization=False):
    """
    Product of two Gaussian mixture

    Input:
    means, covs, weights are list of numpy arrays
    means: length 2 list of (N,d), (M,d) arrays
    covs: length 2 list of (N,d,d), (M,d,d) arrays
    weights: length 2 list of (N,), (M,) arrays

    output
    mean, cov, and weight of the product of two GMMs
    (NxM,d), (NxM,d,d), (NxM) arrays
    """

    mu1, mu2 = means[0], means[1]
    cov1, cov2 = covs[0], covs[1]
    w1, w2 = weights[0], weights[1]

    N, M = w1.shape[0], w2.shape[0]
    d = mu1.shape[1]

    # product of two Gaussian mixtures
    mus = np.empty((N * M, d))
    Sigmas = np.empty((N * M, d, d))
    ws = np.empty((N * M, ))
    for i in range(N):
        for j in range(M):
            mu1i, mu2j = mu1[i], mu2[j]
            cov1i, cov2j = cov1[i], cov2[j]
            w1i, w2j = w1[i], w2[j]
            means = np.vstack((mu1i, mu2j))
            covs = np.stack((cov1i, cov2j))
            muij, covij = gaussian_product(means, covs)
            Cij = ss.multivariate_normal.pdf(mu1i, mu2j, cov1i + cov2j)
            wij = w1i * w2j * Cij
            mus[i * M + j] = muij
            Sigmas[i * M + j] = covij
            ws[i * M + j] = wij

    if normalization:
        # print(ws)
        ws /= ws.sum()

    return (mus, Sigmas, ws)


def GMM_prod(msg_list, normalization=True):
    """
    Product of several Gaussian mixtures: recursion implementation
    msg_list is a list of GMMs

    e.g. msg_list = [(means1, covs1, weights1), (means2, covs2, weights2)]

    """
    msg_list = [msg for msg in msg_list if msg is not None]
    # print(msg_list)
    N = len(msg_list)
    # print(msg_list[0])
    if N == 1:
        return msg_list[0]
    elif N == 2:
        means = [msg_list[0][0], msg_list[1][0]]
        covs = [msg_list[0][1], msg_list[1][1]]
        weights = [msg_list[0][2], msg_list[1][2]]
        return GMM_2prod(means, covs, weights, normalization=normalization)
    elif N > 2:
        means = [msg_list[0][0], msg_list[1][0]]
        covs = [msg_list[0][1], msg_list[1][1]]
        weights = [msg_list[0][2], msg_list[1][2]]
        rmeans, rcovs, rweights = GMM_2prod(means, covs, weights)

        n_msg_list = msg_list[2:]
        n_msg_list.append((rmeans, rcovs, rweights))
        # print(n_msg_list)

        return GMM_prod(n_msg_list, True)


def message_update(pair,
                   local_potential,
                   pair_potential,
                   msg_list,
                   approx=False,
                   K=None,
                   method=None,
                   reg=None):
    """
    Given:
    pair_potential psi_{st} 
    local potential phi_{t}
    msg_list m_{ut} for u in t's neighbour other than s

    return:
    updated message which is also a Gaussian mixture
    self._pair_potential[(s_id, t_id)],
    self._local_potential[t_id], msg_list)
    """
    msg_list = [msg for msg in msg_list if msg is not None]
    t_id, s_id = pair
    potential_means = local_potential[t_id][0]
    potential_covs = local_potential[t_id][1] + pair_potential[(t_id, s_id)]
    potential_weights = local_potential[t_id][2]

    if msg_list == []:
        # Initialization of the message is None
        return (potential_means, potential_covs, potential_weights)
    else:
        #product of neighbor messages
        msg_means, msg_covs, msg_weights = GMM_prod(msg_list, False)

        #product of local and pair potential
        means = []
        covs = []
        weights = []

        for i in range(msg_weights.shape[0]):
            for j in range(potential_weights.shape[0]):
                pmeans = [
                    np.array([local_potential[t_id][0][j]]).reshape((-1, 1)),
                    np.array(
                        (msg_means[i] * potential_covs[j] -
                         potential_means[j] * pair_potential[(t_id, s_id)]) /
                        local_potential[t_id][1][j]).reshape((-1, 1))
                ]
                pcovs = [
                    np.array([potential_covs[j]]).reshape((-1, 1, 1)),
                    np.array(
                        (msg_covs[i] * potential_covs[j] +
                         potential_covs[j] * pair_potential[(t_id, s_id)]) /
                        local_potential[t_id][1][j]).reshape((-1, 1, 1))
                ]

                prod_means, prod_covs, prod_weights = GMM_2prod(
                    pmeans, pcovs, [np.ones(1)] * 2, False)

                means.append(prod_means)
                covs.append(prod_covs)
                weights.append(
                    (prod_weights * msg_weights[i] * potential_weights[j] *
                     (pair_potential[(t_id, s_id)] +
                      local_potential[t_id][1][j]) /
                     local_potential[t_id][1][j]).item())

        # print(weights)

        means = np.vstack(means)
        covs = np.vstack(covs)
        weights = np.array(weights)
        weights /= weights.sum()

        if approx:
            if weights.shape[0] > K:
                rmix = GMR_CTD(means,
                               covs,
                               weights,
                               K,
                               n_pseudo=1,
                               init_method="kmeans",
                               ground_distance=method,
                               reg=reg,
                               max_iter=1000,
                               random_state=0)
                rmix.iterative()
                return (rmix.reduced_means, rmix.reduced_covs,
                        rmix.reduced_weights)
            else:
                return (means, covs, weights)
        else:
            return (means, covs, weights)


def belief_update(node, neighbor_id, local_potential, message):
    """
    update the belief at node i

    The udpated belief is also a GMM
    """
    msg_list = [local_potential[node]]

    for neighbor in neighbor_id:
        msg_list.append(message[(neighbor, node)])

    return GMM_prod(msg_list)


def KDE_rot(samples):
    '''
        https://en.wikipedia.org/wiki/Kernel_density_estimation
    '''
    sigma = np.std(samples, ddof=1)
    return (4 / 3 * sigma**5 / samples.shape[0])**0.2


def gibbs_sampler(pair,
                  local_potential,
                  pair_potential,
                  msg_list,
                  ss=4):
    msg_list = [msg for msg in msg_list if msg is not None]
    t_id, s_id = pair
    potential_means = local_potential[t_id][0]
    potential_covs = local_potential[t_id][1] + pair_potential[(t_id, s_id)]
    potential_weights = local_potential[t_id][2]



    if msg_list == []:
        # Initialization of the message is None
        means, _ = GMM_sampler(potential_means, potential_covs, potential_weights, ss)
        covs = np.tile(KDE_rot(means.reshape(-1,)).reshape((-1, 1, 1)), (ss, 1, 1))
        weights = np.ones((ss, )) / ss
        return (means, covs, weights)
    else:
        #product of neighbor messages
        msg_means, msg_covs, msg_weights = GMM_prod(msg_list, False)

        means, covs, weights = GMM_2prod([msg_means, potential_means],
                                         [msg_covs, potential_covs],
                                         [msg_weights, potential_weights],
                                         True)
        X, _ = GMM_sampler(means, covs, weights, ss)
        covs = np.tile(KDE_rot(X).reshape((-1, 1, 1)), (ss, 1, 1))
        weights = np.ones((ss, )) / ss
        return (means, covs, weights)




class NBP(object):
    def __init__(self,
                 pairs,
                 phi=None,
                 n_iter=4,
                 method="KL",
                 reg=0,
                 K=4,
                 random_state=100):
        """
    pairs: list of lists, e.g.
    [[1, 2], [2, 1], [1, 3], [3, 1], [1, 4], 
    [4, 1], [2, 3], [3, 2], [3, 4], [4, 3]]
    """
        self._n_iter = n_iter
        self._K = K
        self._reg = reg
        self._method = method
        self._seed = random_state

        np.random.seed(random_state)
        if not pairs:
            raise ValueError("Pairs cannot be empty!")
        else:
            self._pairs = pairs
        # print(self._pairs)

        self._nodes = list(set([y for x in self._pairs for y in x]))
        # print(self._nodes)

        # Initialize neighbourhood
        # dictionary by pairs information.
        self._neighbor_dict = {}

        for node in self._nodes:
            indexs = [list(set(t) - set([node])) for t in pairs if node in t]
            self._neighbor_dict[node] = list(
                set([i for sl in indexs for i in sl]))

        # print(self._neighbor_dict)

        # Initialize message to None and variance of pair potential
        self._exact_msg_prev = {}
        self._approx_msg_prev = {}
        self._Gibbs_msg_prev = {}

        self._pair_potential = {}
        for pair in self._pairs:
            self._exact_msg_prev[tuple(pair)] = None
            self._approx_msg_prev[tuple(pair)] = None
            self._Gibbs_msg_prev[tuple(pair)] = None

            self._pair_potential[tuple(pair)] = np.array(
                [1 / phi[tuple(pair)]]).reshape((-1, 1, 1))

        # Initialize local potential, zero mean univariate normal
        self._local_potential = {}
        w = np.random.uniform(0, 1, len(self._nodes))
        for i, node in enumerate(self._nodes):
            self._local_potential[node] = (np.random.uniform(-4, 4, 2).reshape(
                (-1, 1)), np.array([1.0] * 2).reshape(
                    (-1, 1, 1)), np.array([w[i], 1 - w[i]]))
            # self._local_potential[node] = (np.random.uniform(-5,5, 2).reshape(
            #     (-1, 1)), (np.random.uniform(0,1,2)).reshape(
            #         (-1, 1, 1)), np.array([w[i], 1 - w[i]]))
        # print(self._local_potential)

        self._exact_belief = {}
        self._approx_belief = {}
        self._Gibbs_belief = {}
        self._time = []

    def exact_inf(self):
        for i in range(self._n_iter):
            start_time = time.time()
            # update of nonparametric message t -> s
            self._exact_msg_curr = {}
            # print("message update")
            # exact inference
            for pair in self._pairs:
                # print("iter: {0}, pair:".format(i), pair, flush=True)
                t_id = pair[0]  # t node
                s_id = pair[1]  # s node
                t_neighbors = self._neighbor_dict[t_id]
                u_ids = [u for u in t_neighbors if u != s_id]

                # print("t:", t_id, ", s:", s_id)
                # print("u_id:", u_ids)

                # input msg list m(u->t), u is t's neighbor exclude s
                msg_list = []
                if u_ids is not None:
                    for u_id in u_ids:
                        msg_list.append(self._exact_msg_prev[(u_id, t_id)])
                # if t has no neighbor excluding s, then msg_list is empty
                # print(msg_list)

                self._exact_msg_curr[tuple(pair)] = message_update(
                    pair, self._local_potential, self._pair_potential,
                    msg_list)
                # print(self._msg_curr)

            # print(self._msg_prev)
            self._exact_msg_prev = self._exact_msg_curr
            # print(self._msg_prev)

            # belief update
            for node in self._nodes:
                # print(node)
                self._exact_belief[node] = belief_update(
                    node, self._neighbor_dict[node], self._local_potential,
                    self._exact_msg_curr)
            self._time.append(time.time() - start_time)

            output_data = {}
            output_data['message'] = self._exact_msg_prev
            output_data['belief'] = self._exact_belief
            output_data['time'] = self._time

            save_file = 'exact_msg_belief_iter_' + str(i) + 'seed_' + str(
                self._seed) + '.pickle'
            f = open(save_file, 'wb')
            pickle.dump(output_data, f)
            f.close()
            # print(self._belief)

    def approx_inf(self):
        """
        approximate the message when # of component is over k

        """
        for i in range(self._n_iter):
            start_time = time.time()
            # update of nonparametric message t -> s
            self._approx_msg_curr = {}
            # exact inference
            for pair in self._pairs:
                # print("iter: {0}, pair:".format(i), pair, flush=True)
                t_id = pair[0]  # t node
                s_id = pair[1]  # s node
                t_neighbors = self._neighbor_dict[t_id]
                u_ids = [u for u in t_neighbors if u != s_id]

                # print("t:", t_id, ", s:", s_id)
                # print("u_id:", u_ids)

                # input msg list m(u->t), u is t's neighbor exclude s
                msg_list = []
                if u_ids is not None:
                    for u_id in u_ids:
                        msg_list.append(self._approx_msg_prev[(u_id, t_id)])
                # if t has no neighbor excluding s, then msg_list is empty
                # print(msg_list)

                self._approx_msg_curr[tuple(pair)] = message_update(
                    pair, self._local_potential, self._pair_potential,
                    msg_list, True, self._K, self._method, self._reg)
                # print(self._msg_curr)

            # print(self._approx_msg_prev)
            self._approx_msg_prev = self._approx_msg_curr
            # print(self._msg_prev)

            # belief update
            for node in self._nodes:
                # print(node)
                self._approx_belief[node] = belief_update(
                    node, self._neighbor_dict[node], self._local_potential,
                    self._approx_msg_curr)

            self._time.append(time.time() - start_time)

            output_data = {}
            output_data['message'] = self._approx_msg_prev
            output_data['belief'] = self._approx_belief
            output_data['time'] = self._time

            save_file = 'approx_msg_belief_iter' + str(i) + 'seed_' + str(
                self._seed) + 'gd_' + self._method + 'K_' + str(
                    self._K) + 'reg_' + str(self._reg) + '.pickle'
            f = open(save_file, 'wb')
            pickle.dump(output_data, f)
            f.close()

    def Gibbs_inf(self):
        for i in range(self._n_iter):
            start_time = time.time()
            # update of nonparametric message t -> s
            self._Gibbs_msg_curr = {}
            # print("message update")
            # exact inference
            for pair in self._pairs:
                # print("iter: {0}, pair:".format(i), pair, flush=True)
                t_id = pair[0]  # t node
                s_id = pair[1]  # s node
                t_neighbors = self._neighbor_dict[t_id]
                u_ids = [u for u in t_neighbors if u != s_id]

                # print("t:", t_id, ", s:", s_id)
                # print("u_id:", u_ids)

                # input msg list m(u->t), u is t's neighbor exclude s
                msg_list = []
                if u_ids is not None:
                    for u_id in u_ids:
                        msg_list.append(self._Gibbs_msg_prev[(u_id, t_id)])
                # if t has no neighbor excluding s, then msg_list is empty
                # print(msg_list)

                self._Gibbs_msg_curr[tuple(pair)] = gibbs_sampler(
                    pair, self._local_potential, self._pair_potential,
                    msg_list)
                # print(self._msg_curr)

            # print(self._msg_prev)
            self._Gibbs_msg_prev = self._Gibbs_msg_curr
            # print(self._msg_prev)

            # belief update
            for node in self._nodes:
                print("belief update")
                # print(node)
                self._Gibbs_belief[node] = belief_update(
                    node, self._neighbor_dict[node], self._local_potential,
                    self._Gibbs_msg_curr)
                print(node, self._Gibbs_belief[node][0].shape[0])
            print("iter:", i, time.time() - start_time)

            output_data = {}
            output_data['message'] = self._exact_msg_prev
            output_data['belief'] = self._exact_belief

            save_file = 'Gibbs_msg_belief_iter_' + str(i) + 'seed_' + str(
                self._seed) + '.pickle'
            f = open(save_file, 'wb')
            pickle.dump(output_data, f)
            f.close()
            # print(self._belief)


    

def main(seed, phi, pairs):
    nbp = NBP(pairs, phi, K=4, random_state=seed)
    nbp.exact_inf()
    

    nbp = NBP(pairs, phi, K=4, method="KL", reg=0, random_state=seed)
    nbp.approx_inf()
    
    nbp = NBP(pairs, phi, K=4, method="KL", reg=1, random_state=seed)
    nbp.approx_inf()

    nbp = NBP(pairs, phi, K=4, method="WKL", reg=0, random_state=seed)
    nbp.approx_inf()

    nbp = NBP(pairs, phi, K=4, method="WKL", reg=1, random_state=seed)
    nbp.approx_inf()

    nbp = NBP(pairs, phi, K=4, method="W2", reg=0, random_state=seed)
    nbp.approx_inf()

    nbp = NBP(pairs, phi, K=4, method="W2", reg=1, random_state=seed)
    nbp.approx_inf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset split GMM estimator comparison')
    parser.add_argument('--seed', type=int, default=1, help='index of repetition')
   
    
    args= parser.parse_args()
    seed = args.seed
    phi = np.array(
        [1, 0.2, 0.4, 0.6, 0.2, 1, 0.01, 0, 0.4, 0.01, 1, 0.8, 0.6, 0, 0.8,
         1]).reshape((4, 4))
    pairs = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [1, 2], [2, 1],
             [2, 3], [3, 2]]

    main(seed, phi, pairs)