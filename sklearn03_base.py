 numpy as np

from sklearn.base import BaseEstimator

class MyEstimator(BaseEstimator):
    def __init__(self, *, param=1):
        self.param = param
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    def predict(self, X):
        return np.full(shape=X.shape[0], fill_value=self.param)

estimator = MyEstimator(param=2)

estimator.get_params()
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([1, 0, 1])
estimator.fit(X, y).predict(X)
estimator.set_params(param=3).fit(X, y).predict(X)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score
