from src.core.estimators import OLS
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self, memory=2, estimator_factory=OLS):
        self.memory = memory
        self._estimator_real = estimator_factory()
        self._estimator_imag = estimator_factory()

    @abstractmethod
    def build_regressors(self,x, y=None):
        ...

    def align(self, y):
        y = y[self.memory:]
        return y

    def fit(self, dataset):
        Phi = self.build_regressors(dataset.x, dataset.y)
        y = self.align(dataset.y)
        self._estimator_real.fit(Phi, y.real)
        self._estimator_imag.fit(Phi, y.imag)
        return self

    def predict(self, dataset):
        Phi = self.build_regressors(dataset.x, dataset.y)
        yreal = self._estimator_real.predict(Phi)
        yimag = self._estimator_imag.predict(Phi)

        y = yreal + 1j * yimag
        return y
