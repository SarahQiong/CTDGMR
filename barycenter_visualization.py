import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from matplotlib.collections import EllipseCollection


def get_ww_hh_an(covar):
    U, s, Vt = np.linalg.svd(covar)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = np.sqrt(s) 
    # width, height = np.sqrt(s) / np.sqrt(s).sum()
    return width, height, angle


def Gaussian_barycenter(covs, weights=None, tol=1e-5, ground_distance="W2"):
    """Compute the Wasserstein or KL barycenter of zero mean Gaussian measures.

    Parameters
    ----------
    covs :  array-like, shape (n, d, d)
    weights : array-like, shape (n,)
        The weight in front of the Wasserstein distance.
    ground_distance: string. One of "W2" and "KL"

    Returns
    -------
    mean and covariance of the Gaussian Wasserstein barycenter.

    """
    m, d, _ = covs.shape
    if weights is None:
        weights = np.ones((m, 1)) / m
    else:
        # weight standardization
        weights = weights / weights.sum()

    if ground_distance == "KL":
        barycenter_covs = np.sum(covs * weights.reshape((-1, 1, 1)), axis=0)

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
    return barycenter_covs


def barycenter_visualization(covs, case, ratio):
    Q = 11
    t = np.linspace(0, 1, Q)
    X, Y = np.meshgrid(t, t)
    XY = np.column_stack((X.ravel(), Y.ravel()))

    Lambda = np.column_stack(
        ((1 - X.ravel()) * (1 - Y.ravel()), (1 - X.ravel()) * Y.ravel(),
         X.ravel() * (1 - Y.ravel()), X.ravel() * Y.ravel()))


    W_width, W_height, W_angle = np.zeros((Lambda.shape[0], )), np.zeros(
        (Lambda.shape[0], )), np.zeros((Lambda.shape[0], ))
    KL_width, KL_height, KL_angle = np.zeros((Lambda.shape[0], )), np.zeros(
        (Lambda.shape[0], )), np.zeros((Lambda.shape[0], ))


    for i, l in enumerate(Lambda):
        W_barycenter = Gaussian_barycenter(covs, weights=l, ground_distance='W2')
        W_width[i], W_height[i], W_angle[i] = get_ww_hh_an(W_barycenter)
        KL_barycenter = Gaussian_barycenter(covs, weights=l, ground_distance='KL')
        KL_width[i], KL_height[i], KL_angle[i] = get_ww_hh_an(KL_barycenter)


    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ec = EllipseCollection(ratio * W_width.reshape((Q, Q)),
                           ratio * W_height.reshape((Q, Q)),
                           W_angle.reshape((Q, Q)),
                           units='x',
                           offsets=XY,
                           transOffset=ax.transData)
    ec.set_array((X + Y).ravel())
    ax.add_collection(ec)
    ax.autoscale_view()
    ax.axis('off')
    plt.savefig('./W_barycenter_'+str(case)+ '.png', dpi=200, bbox_inches='tight')

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ec = EllipseCollection(ratio * KL_width.reshape((Q, Q)),
                           ratio * KL_height.reshape((Q, Q)),
                           KL_angle.reshape((Q, Q)),
                           units='x',
                           offsets=XY,
                           transOffset=ax.transData)
    ec.set_array((X + Y).ravel())
    ax.add_collection(ec)
    ax.autoscale_view()
    ax.axis('off')
    plt.savefig('./KL_barycenter_'+str(case)+ '.png', dpi=200, bbox_inches='tight')


# Plot for comparing the barycenter of four covariance matrices
K, d = 4, 2
covs = np.zeros((K, d, d))

covs[0] = np.array([0.9802, 0.0973, 0.0973, 0.0198]).reshape((2, 2))
covs[1] = np.array([0.4275, 0.5249, 0.5249, 0.6725]).reshape((2, 2))
covs[2] = np.array([0.7572, -0.3164, -0.3164, 0.1428]).reshape((2, 2))
covs[3] = np.array([0.5, 0, 0, 0.5]).reshape((2, 2))

barycenter_visualization(covs, 1, 0.12)


covs = np.zeros((K, d, d))
covs[0] = np.array([0.8430, 0.0695, 0.0695, 0.1570]).reshape((2, 2))
covs[1] = np.array([0.4625, 0.3749, 0.3749, 0.6375]).reshape((2, 2))
covs[2] = np.array([0.6695, -0.2260, -0.2260, 0.2305]).reshape((2, 2))
covs[3] = np.array([0.5, 0, 0, 0.5]).reshape((2, 2))

barycenter_visualization(covs, 2, 0.12)