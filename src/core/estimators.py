from abc import ABC, abstractmethod
import numpy as np

class Estimator(ABC):
    def __init__(self):
        self.theta = None

    @abstractmethod
    def fit(self, Phi, y):
        ...

    def predict(self, Phi):
        y_pred = Phi @ self.theta
        return y_pred

class OLS(Estimator):
    def __init__(self):
        super().__init__()

    def fit(self, Phi, y):
        self.theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        return self
