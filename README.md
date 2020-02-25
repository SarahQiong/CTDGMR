This repository contains the code for the paper

**[A Unified Framework for Gaussian Mixture Reduction with Composite Transportation Distance](https://arxiv.org/abs/2002.08410.pdf)**


Function usage
```
b_means = np.array([1,1]*2+[-1,1]*2+[-1,-1]*2+[1,-1]*2).reshape((-1,2))
b_covs = np.array([1,0,0,0.01,0.01,0,0,1]).reshape((-1,2,2))
b_covs = np.tile(b_covs, (4,1,1))
K, d = b_means.shape
b_weights = np.ones((K,))/K

rmix = GMR_CTD(b_means, b_covs, b_weights, 4, n_pseudo=1, init_method=init,
               ground_distance=dist, reg=0.1, max_iter=1000, random_state=0)
rmix.iterative()
r_means, r_covs, r_weights = rmix.reduced_means, rmix.reduced_covs, rmix.reduced_weights

```
Arguments of the function
```
b_means: np.array, shape (N,d)
b_covs: np.array, shape (N,d,d)
b_weights: np.array, shape (N,)
init_method: "Salmond", "Salmond", "Runnalls", "Williams", "W", "kmeans", "user"
ground_distance: "W2","KL","WKL"
```


