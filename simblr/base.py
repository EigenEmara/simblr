import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class BaseModel:
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
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        phi = self.poly_features.fit_transform(x)
        return phi @ self.params

    @property
    def theta(self) -> np.ndarray:
        return self.params
