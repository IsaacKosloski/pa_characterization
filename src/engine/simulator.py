import numpy as np
import pandas as pd
from src.core.dataset import Dataset


class Simulator:
    def __init__(self, model):
        self.model = model
        if model.coefficients()["real"] is None:
            raise ValueError("modelo não foi ajustado (chame o fit antes)")

    def run(self, x):
        return self.model.predict(Dataset(x, np.zeros_like(x)))

    def to_csv(self, path, x, y_true=None):
        eps = 1e-12
        y_hat = self.run(x)
        xa = self.model.align(x)
        ya = self.model.align(y_hat)
        dados = {
            "Xreal": xa.real, "Ximg": xa.imag, "X_magnitude": np.abs(xa),
            "Ypred_real": ya.real, "Ypred_img": ya.imag,
        }
        if y_true is not None:
            yt = self.model.align(y_true)
            G = yt / (xa + eps)                       # ganho complexo do PA por amostra
            dados.update({
                "Yreal": yt.real, "Yimg": yt.imag,
                "error_point": np.abs(yt - ya),        # erro por amostra
                "G_real": G.real, "G_img": G.imag,
                "G_magnitude": np.abs(G),
                "G_magnitude_dB": 20 * np.log10(np.abs(G) + eps),
                "G_phase_deg": np.angle(G, deg=True),
            })
        pd.DataFrame(dados).to_csv(path, index=False)
