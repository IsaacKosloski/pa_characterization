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
    """θ = (ΦᵀΦ)⁻¹Φᵀy"""
    def __init__(self):
        super().__init__()

    def fit(self, Phi, y):
        self.theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        return self

class Ridge(Estimator):
    """θ = (ΦᵀΦ + λI)⁻¹Φᵀy"""
    def __init__(self, lam=1e-3):
        super().__init__()
        self.lam = lam

    def fit(self, Phi, y):
        k = Phi.shape[1]
        A = Phi.T @ Phi + self.lam * np.eye(k)
        b = Phi.T @ y
        self.theta = np.linalg.solve(A, b)
        return self
