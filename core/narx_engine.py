"""
Motor de inferência do modelo NARX.

Responsável por receber X[n] (sinal de entrada complexo) e
produzir Ŷ[n] (sinal de saída predito complexo), executando
a equação NARX completa com memória autorregressiva e
regressores polinomiais da entrada exógena.

Equação implementada:
    Y(t) = c
         + A1 * Y(t-1) + A2 * Y(t-2)
         + B * Φ(X(t), X(t-1), X(t-2))

Onde Φ é o vetor de regressores exógenos (linear + quadrático + cúbico).
"""

import numpy as np
from typing import Tuple
from core.model_config import NARXModelConfig

class NARXEngine:
    """
    Executa a predição amostra-a-amostra do modelo NARX.

    Parameters
        config: NARXModelConfig
            Configuração completa do modelo (interceptos, A1, A2, betas).
    Use
        engine = NARXEngine(config)
        Y_pred = engion.predict(X_complex)
    """

    def __init__(self, config: NARXModelConfig):
        self.config = config
        # Pré-extrair arrays para velocidade no loop de predição
        self._c = config.intercepts.to_array()      # shape(2,)
        self._A1 = config.A1.to_matrix()  # shape (2,2)
        self._A2 = config.A2.to_matrix()  # shape (2,2)
        self._b = config.betas  # objeto ExogenousBetas, com atributos

    # --- Construção dos Regressores Exógenos
    @staticmethod
    def _build_regressors(
            xr0: float, xi0: float,  # X(t): real e imaginário
            xr1: float, xi1: float,  # X(t-1)
            xr2: float, xi2: float,  # X(t-2)
    ) -> np.ndarray:
        """
        Monta o vetor de regressores phi(X) para um instante t.

        Ordem (18 regressores para cada saída = 36 total, no entanto, os betas são separados por saída, daí retornar 18):
            [Xr, Xi,                     deg 1 (lag 0)
             Xr_l1, Xi_l1,               deg 1 (lag 1)
             Xr_l2, Xi_l2,               deg 1 (lag 2)
             Xr^2, Xr_l1^2, Xr_l2^2,     deg 2 (real)
             Xi^2, Xi_l1^2, Xi_l2^2,     deg 2 (img)
             Xr^3, Xr_l1^3, Xr_l2^3,     deg 3 (real)
             Xi^3, Xi_l1^3, Xi_l2^3]     deg 3 (img)
        :return: np.array[] com todos os 18 termos
        """
        return np.array([
            # -- Linear
            xr0, xi0, xr1, xi1, xr2, xi2,
            # -- Quadrático
            xr0**2, xi0**2, xr1**2, xi1**2, xr2**2, xi2**2,
            # -- Cúbico
            xr0**3, xi0**3, xr1**3, xi1**3, xr2**3, xi2**3,
        ])

    def _beta_vector_yreal(self) -> np.ndarray:
        """
        Vetor de betas na mesma ordem que _build_regressors() para a saída Yreal.
        :return: np.array[] com todos os 18 termos
        """
        b = self._b
        return np.array([
            b.Xreal_Yreal, b.Ximg_Yreal,
            b.Xreal_lag1_Yreal, b.Ximg_lag1_Yreal,
            b.Xreal_lag2_Yreal, b.Ximg_lag2_Yreal,
            b.Xreal_deg2_Yreal, b.Xreal_lag1_deg2_Yreal, b.Xreal_lag2_deg2_Yreal,
            b.Ximg_deg2_Yreal, b.Ximg_lag1_deg2_Yreal, b.Ximg_lag2_deg2_Yreal,
            b.Xreal_deg3_Yreal, b.Xreal_lag1_deg3_Yreal, b.Xreal_lag2_deg3_Yreal,
            b.Ximg_deg3_Yreal, b.Ximg_lag1_deg3_Yreal, b.Ximg_lag2_deg3_Yreal,
        ])

    def _beta_vector_yimg(self) -> np.ndarray:
        """
        Vetor de betas na mesma ordem que _build_regressors() para a saída Yimg.
        """
        b = self._b
        return np.array([
            b.Xreal_Yimg, b.Ximg_Yimg,
            b.Xreal_lag1_Yimg, b.Ximg_lag1_Yimg,
            b.Xreal_lag2_Yimg, b.Ximg_lag2_Yimg,
            b.Xreal_deg2_Yimg, b.Xreal_lag1_deg2_Yimg, b.Xreal_lag2_deg2_Yimg,
            b.Ximg_deg2_Yimg, b.Ximg_lag1_deg2_Yimg, b.Ximg_lag2_deg2_Yimg,
            b.Xreal_deg3_Yimg, b.Xreal_lag1_deg3_Yimg, b.Xreal_lag2_deg3_Yimg,
            b.Ximg_deg3_Yimg, b.Ximg_lag1_deg3_Yimg, b.Ximg_lag2_deg3_Yimg,
        ])

    # --- Predição Vetorizada
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz a saída Y[n] do PA para a sequência de entrada X[n].

        :param
            X : np.ndarray, dtype complex, shape (N,) Sinal de entrada complexo. Representa a envoltória complexa (I/Q) do sinal de RF após down-conversion.

        :return
            Y_pred : np.ndarray, dtype complex, shape (N,) Sinal de saída predito pelo modelo NARX.
        """
        N = len(X)
        Xr = X.real
        Xi = X.imag

        # Buffer de saída (inicializado com zeros - condições iniciais)
        Y_pred = np.zeros(N, dtype=complex)

        # Pré-calcular vetores de betas (constantes durante a predição)
        bv_real = self._beta_vector_yreal()
        bv_img  = self._beta_vector_yimg()

        # A1 e A2 ds config (transposta para multiplicação por vetor coluna)
        A1 = self._A1   # shape (2,2)
        A2 = self._A2   # shape (2,2)
        c  = self._c    # shape (2,)

        for t in range(N):
            # -- Regressores de X (com padding zero para lags iniciais)
            xr0 = Xr[t]
            xi0 = Xi[t]
            xr1 = Xr[t-1] if t >= 1 else 0.0
            xi1 = Xi[t-1] if t >= 1 else 0.0
            xr2 = Xr[t-2] if t >= 2 else 0.0
            xi2 = Xi[t-2] if t >= 2 else 0.0

            phi = self._build_regressors(xr0, xi0, xr1, xi1, xr2, xi2)

            # -- Parte exógena: B * phi(X)
            exog_real = np.dot(bv_real, phi)
            exog_img = np.dot(bv_img, phi)

            # -- Parte autoregressiva: A1*Y(t-1) + A2*Y(t-2)
            Y_prev1 = np.array([Y_pred[t - 1].real, Y_pred[t - 1].imag]) if t >= 1 else np.zeros(2)
            Y_prev2 = np.array([Y_pred[t - 2].real, Y_pred[t - 2].imag]) if t >= 2 else np.zeros(2)

            ar_contrib = A1 @ Y_prev1 + A2 @ Y_prev2

            # -- Saída final
            yreal_t = c[0] + ar_contrib[0] + exog_real
            yimg_t = c[1] + ar_contrib[1] + exog_img

            Y_pred[t] = complex(yreal_t, yimg_t)

        return Y_pred

    # --- Métricas
    @staticmethod
    def rmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        RMSE sobre a envoltória complexa. Calcula o erro em ambas as componentes (I e Q) simultaneamente.
            RMSE = sqrt(mean(|Y_true - Y_pred|^2))

        :param Y_true:
        :param Y_pred:
        :return:  float : valor escalar do RMSE (mesma unidade do sinal).
        """
        err = Y_true - Y_pred
        return float(np.sqrt(np.mean(np.abs(err) ** 2)))

    @staticmethod
    def nmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        NMSE (Normalized Mean Square Error) em dB. Métrica padrão em modelagem de PAs.
            NMSE_dB = 10 * log10( E[|e|²] / E[|Y_true|²] )

        Valores típicos:
            < -30 dB → excelente
            -30 a -20 dB → bom
            > -20 dB → necessita melhoria
        """
        mse = np.mean(np.abs(Y_true - Y_pred) ** 2)
        power = np.mean(np.abs(Y_true) ** 2)
        return float(10.0 * np.log10(mse / (power + 1e-15)))

def compute_gain(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calcula o ganho complexo G[n] = Y[n] / X[n].

    Usado para verificar a linearidade:
        - PA linear ideal → |G[n]| constante, ∠G[n] constante.
        - PA não-linear → |G[n]| varia com |X[n]| (AM-AM)
                           ∠G[n] varia com |X[n]| (AM-PM).

    :param
    X : np.ndarray complex — sinal de entrada
    Y : np.ndarray complex — sinal de saída

    :return
    G : np.ndarray complex — ganho complexo por amostra
    """
    # Evita divisão por zero (silencia amostras com X ≈ 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(np.abs(X) > 1e-10, Y / X, complex(0, 0))
    return G
