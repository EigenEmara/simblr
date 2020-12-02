import numpy as np
import scipy.linalg

from .base import BaseModel

'''
Maximum likelihood estimation (MLE)
-----------------------------------
For a linear regression setting, we are given data set (observations) {X, Y} = {(x1, y1), ... , (x_N, y_N)}
where x is a D dimensional vector and an element of R^D vector space and y is a scalar.

Each input y corresponds to a noisy observation of a true function f(x), such that:
y = f(x) + ϵ, where ϵ is a random variable the represents noise, and we assume that ϵ is Gaussian N(0, σ^2).

According to Deisenroth et al. (2020), due to the observation being noisy, we are going to adopt a probabilistic
apporach and model noise using a likelihood function, such that the distribution of our target variable is:
p(y|x) = N(f(x), σ^2)
f(x) = x.T @ θ
p(y|x,θ) = N(x.T @ θ, σ^2)

In, MLE we assume that each input from the data set is i.i.d (independent and identically distributed),
so one can write for all the observations, the likelihood function is:
p(Y|X)  = p(y_1, y_2, ... , y_N|x_1, x_2, ..., x_N)
        = p(y_1|x_1) * p(y_2|x_2) * ... * p(y_N, x_N)
        = N(y_1|x_1.T @ θ) * N(y_2|x_2.T @ θ) * ... * N(y_N|x_N.T @ θ)

Our goal is to minimize the negative log of the likelihood function, where we obtain optimum θ*:
θ* = inv(X.T @ X) @ X.T @ Y

And in case of polynomial regression with feature matrix Φ
θ* = inv(Φ.T @ Φ) @ Φ.T @ y

And if we minimized the negative log likelihood w.r.t σ^2, we can obtain the maximum likelihood estimate
of the noise variance:
h = Φ.T @ θ
σ^2 = 1/N * ((y - h).T @ (y - h))

It should be noted that maximum likelihood is prone to overfitting,
and it's basically a least square optimization problem without regression.
'''
class MaxLikelihood(BaseModel):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, M=4):
        super().__init__(x_train, y_train, M)

    def fit(self):
        ''' Calculates optimum MLE parameters

        Returns
        -------
        params: ndarray of shape ((n_features * M) + 1, 1)
            Optimum maximum likelihood estimation parameters vector.
        '''
        # github.com/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_linear_regression.solution.ipynb
        # scicomp.stackexchange.com/questions/36342/advantage-of-diagonal-jitter-for-numerical-stability
        kappa = 1e-08
        D = self.phi.shape[1]

        Pt = self.phi.T @ self.y_train
        PP = self.phi.T @ self.phi + kappa*np.eye(D)
        theta_MLE = scipy.linalg.solve(PP, Pt)

        self.params = theta_MLE

        return self.params

