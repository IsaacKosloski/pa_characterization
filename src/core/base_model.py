from src.core.estimators import OLS
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self, memory=2, estimator_factory=OLS):
        self.memory = memory

        self.trim_front = memory
        self.trim_back = 0

        self._estimator_real = estimator_factory()
        self._estimator_imag = estimator_factory()

    @abstractmethod
    def build_regressors(self,x, y=None):
        ...

    @staticmethod
    def _shift(arr, s):
        N = len(arr)

        # Para o passado
        if s > 0:
            zeros = np.zeros(s, dtype=arr.dtype)
            slice_arr = arr[: N - s]
            return np.concatenate((zeros, slice_arr))
        # Para o futuro
        if s < 0:
            zeros = np.zeros(-s, dtype=arr.dtype)
            slice_arr = arr[-s:]
            return np.concatenate((slice_arr, zeros))
        else:
            return arr.copy()

    def align(self, arr):
        return arr[self.trim_front: len(arr) - self.trim_back]

    def fit(self, dataset):
        Phi = self.build_regressors(dataset.x, dataset.y)
        Phi_train = self.align(Phi)
        y_train = self.align(dataset.y)

        self._estimator_real.fit(Phi_train, y_train.real)
        self._estimator_imag.fit(Phi_train, y_train.imag)
        return self

    def predict(self, dataset):
        Phi = self.build_regressors(dataset.x, dataset.y)
        yreal = self._estimator_real.predict(Phi)
        yimag = self._estimator_imag.predict(Phi)

        y = yreal + 1j * yimag
        return y

    def evaluate(self, dataset, metric):
        y_true = self.align(dataset.y)
        y_pred =self.align(self.predict(dataset))
        return metric.compute(y_true, y_pred)

    def coefficients(self):
        coeff = {"real": self._estimator_real.theta, "imag": self._estimator_imag.theta}
        return coeff

