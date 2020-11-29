import numpy as np
import scipy.linalg

from .base import BaseModel

'''
Maximum A Posteriori Estimation
-------------------------------
As per Deisenrot et al. (2020), maximum likelihood estimation is prone to overfitting,
when we run into over fitting the parameters values become relatively large.

In order to mitigate the effect of overfitting, we place a Gaussian prior on the parameters.
For example, when we place p(θ) = N(0, 1) on a single parameter, we expect the parameter to be
in the interval [-2,2].

So, we have a data set (X, Y), we would like to seek parameters that maximize the posterior
distribution p(θ|X, Y)

p(θ|X, Y) ∝ p(Y|X, θ) (likelihood) * p(θ) (prior)

So, instead of maximizing likelihood, we maximize the posterior that gives us the optimal
parameter vector θ, given the data set and prior distribution of θ.

p(θ) = N(0, b^2 I)

For, a polynomial feature matrix phi (check likelihood.py documentations):
θ_MAP = inv(phi.T @ phi + (σ^2/b^2) I) @ phi.T @ Y

MAP can be considered as a least square optimization with l2 regularization.
'''


class MaxAPosteriori(BaseModel):
    noise_var = 1.0
    prior_var = 0.25

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, M=4, beta=4.0):
        self.beta = beta  # noise_var/prior_var (i.e.: Regularization coefficient)
        super().__init__(x_train, y_train, M)

    def fit(self):
        D = self.phi.shape[1]
        PP = (self.phi.T @ self.phi) + (self.beta ** 2 * np.eye(D))
        theta_MAP = scipy.linalg.solve(PP, self.phi.T @ self.y_train)

        self.params = theta_MAP

        return self.params
