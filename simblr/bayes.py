import numpy as np
import scipy.linalg

from .base import BaseModel

'''
Bayes Linear Regression
-----------------------
From Deisenroth et al. (2020):
"Bayesian linear regression pushes the idea of the parameter prior a step regression further and does not even 
attempt to compute a point estimate of the parameters, but instead the full posterior distribution over the parameters
is taken into account when making predictions. This means we do not fit any parameters, 
but we compute a mean over all plausible parameters settings (according to the posterior)."

We start by placing a prior on θ: p(θ) = N(m0, s0)
our likelihood function: p(y|x, θ) = N(y|Phi.T @ θ, σ^2)

We have many parameters to fit out model, all sampled from the prior distribution p(θ),
with a mean parameter vector m0 and distribution variance σ^2.

---------------------------
I - PRIOR BASED PREDICTIONS
---------------------------
We can make predictions by averaging out all plausible parameter settings, such that:
p(y*|x*) = ∫p(y*|x,θ) p(θ) dθ  
         = E[p(y*|x,θ)] 
         = N(phi*.T @ m0, phi*.T @ s0 @phi* + σ^2)

And if we are interested in predicting noise-free function values f(x*) (we just omit σ^2):
p(y*|x*) = N(phi*.T @ m0, phi*.T @ s0 @phi*)

We can have a parameter posterior, given training data set X, Y, 
we we just need to replace the prior p(θ) with the posterior p(θ|X, Y).

---------------------------
II - POSTERIOR DISTRIBUTION
---------------------------
p(θ|X, Y) ∝ p(Y|X,θ)p(θ) = N(θ|mN, sN)
sN = inv(s0 + σ^-2 @ phi.T @ phi)
mN = sN @ (inv(s0) @ m0 + σ^-2 phi.T @ Y)

--------------------------
III - POSTERIOR PREDICTION
--------------------------
p(y*|X, Y, x*)  = ∫ p(y*|x*, θ) p(θ|X, Y) dθ
                = N(y*| phi*.T @ mN, phi*.T @ sN @ phi* + σ^2)
                
"the Bayesian linear regression model additionally tells us that the posterior uncertainty is huge. 
This information can be critical when we use these predictions in a decision-making system, 
where bad decisions can have significant consequences
'''


class BayesLinearRegression(BaseModel):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, M=4):
        self.mN = None
        self.sN = None

        self.alpha = 0.25
        self.sigma = 1.0
        super().__init__(x_train, y_train, M=M)

    def fit(self):
        m0 = np.zeros((self.M + 1, 1))
        s0 = np.eye(self.M + 1) * self.alpha
        sN = np.linalg.inv(((self.phi.T @ self.phi) * self.sigma ** -2) + np.linalg.inv(s0))
        mN = sN @ ((self.phi.T @ self.y_train) * self.sigma ** -2) + (np.linalg.inv(s0) @ m0)

        self.sN = sN
        self.mN = mN

        return None

    def predict_x_star(self, x_star):
        phi_star = self.poly_features.fit_transform(
            np.array([x_star]).reshape(1, -1)).reshape(self.M + 1, 1)
        mu = phi_star.T @ self.mN
        marg_var = phi_star.T @ self.sN @ phi_star
        var = marg_var + self.sigma

        return float(mu), float(marg_var), float(var)

    def predict(self, x: np.ndarray):
        pred = np.vectorize(self.predict_x_star)
        mean, marginal_variance, variance = pred(x)
        return mean, marginal_variance, variance
    
    @property
    def theta(self):
        raise NotImplementedError
