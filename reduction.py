import os
import pickle
import pyreadr
import argparse
import numpy as np
from CTDGMR.minCTD import GMR_CTD
from CTDGMR.optGMR import GMR_opt_BFGS
# stop the warning message in initialization
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Demo for Gaussian mixture reduction to reduce mixture of order N to order K in d dimensional space
N =  10
d =  2

# Set the value for original mixture 
# The parameter values are randomly generated for domenstration purposes
np.random.seed(1)
base_means = np.random.randn(N*d).reshape((N, d)) # float array with dimension (N, d)
base_covs = np.empty((N, d, d)) 
for i in range(N):
    base_cov = np.random.randn(d, d)
    base_cov = base_cov.dot(base_cov.T)
    base_cov += 0.5 * np.eye(d)
    base_covs[i] = base_cov

# float array with dimension (N, d, d)
base_weights = np.random.uniform(0, 1, N) 
base_weights /= base_weights.sum() # float array with dimension (N, )

# Set the order of the reduced mixture
K = 5 
#####################
# perform reduction
#####################

# Approach 1: minimium ISE in Williams 2006
obj = np.Inf
for i in range(1): # speiciy the number of initial values by kmeans
    reduced_mix = GMR_opt_BFGS(base_means,
                               base_covs,
                               base_weights,
                               K,
                               loss='ISE',
                               init_method='kmeans',
                               tol=1e-8,
                               random_state=i)
    reduced_mix.run()
    if reduced_mix.obj < obj:
        reduction = reduced_mix
        obj = reduced_mix.obj

# Approach 2: Reduction by our proposed CTD based approach

cost_function = 'KL' # cost function in CTD
reg = 0.0 # strength of lambda in the regularized CTD
obj = np.Inf
for i in range(5): # speiciy the number of initial values by kmeans
    reduced_mix = GMR_CTD(base_means,
                          base_covs,
                          base_weights,
                          K,
                          n_pseudo=1,
                          init_method='kmeans',
                          ground_distance=cost_function,
                          reg=reg,
                          max_iter=1000,
                          tol=1e-5,
                          random_state=i)
    reduced_mix.iterative()
    if reduced_mix.obj < obj:
        reduction = reduced_mix
        obj = reduced_mix.obj

# Get the parameter values of the reduced mixture
reduced_means = reduction.reduced_means
reduced_covs = reduction.reduced_covs
reduced_weights = reduction.reduced_weights

print(reduced_means)


