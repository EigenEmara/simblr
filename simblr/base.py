import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class BaseModel:
    """ Base class for all linear regression models.

    BaseModel provides basic methods and members that are shared 
    between all other models and overriden such as MaxLikelihood and MaxAPosteriori.

    BaseModel cannot be used directly and only intended to be inherited.


    Parameters
    ----------
    x_train: ndarray of shape (n_examples, n_features)
        Training input vector. Copied to BaseModel.x_train

    y_train: ndarray of shape (n_examples, 1)
        Target variable vector. Copied to BaseModel.y_train

    M: int
        Polynomial degree of linear regression function. 
        Copied to BaseModel.M
    
    Attributes
    ----------
    poly_features: sklearn.preprocessing.PolynomialFeatures
        an instance of scikit's PolynomialFeatures with polynomial degree = M
    
    phi: ndarray of shape (n_examples, (n_features * M) + 1)
        Polynomial feature matrix (Design matrix).
    
    params: ndarray of shape ((n_features * M) + 1, 1)
        Optimum model parameters, only used in maximum likelihood estimation
        and maximize a posteriori.
    
    x_train: ndarray of shape (n_examples, n_features)
        (Check 'Parameters' section)

    y_train: ndarray of shape (n_examples, 1)
        (Check 'Parameters' section)

    M: int
        (Check 'Parameters' section)

    """
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, M=4):
        self.x_train = x_train
        self.y_train = y_train
        self.M = M

        self.m, self.n = self.x_train.shape
        self.poly_features = PolynomialFeatures(M)
        self.phi = self.poly_features.fit_transform(self.x_train)

        # Optimum model parameter
        self.params = None
        self.fit()

    def fit(self):
        """ Fits our model using training data 
        
        Returns
        -------
        params: ndarray of shape ((n_features * M) + 1, 1)
            Optimum parameters for a linear regression model (specifically for MLE and MAP models).

            NOTE:   In the case of BayesLinearRegression fit() returns None.
                    because  we do not fit any parameters, 
                    but we calculate "a mean over all plausible parameters settings"
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predicts output given an input vector x.

        Parameters
        ----------
        x: ndarray of shape (m, 1)
            Input vector to calculate corresponding regression output.
        
        Returns
        -------
        y: ndarray of shape (m, 1)
            Output of the linear model.

        """
        phi = self.poly_features.fit_transform(x)
        y = phi @ self.params
        return y

    @property
    def theta(self) -> np.ndarray:
        """ Provides vector of optimum model parameters for MLE and MAP.

        Returns
        -------
        params: ndarray of shape ((n_features * M) + 1, 1)
            A vector of optimum model parameters.

        """
        return self.params
