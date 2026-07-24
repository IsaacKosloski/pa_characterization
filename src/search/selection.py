import numpy as np


def _neg2logL(rmse, N):
    return N * np.log(rmse ** 2)  # = N·ln(SSE/N), com SSE = rmse²·N


def aic(rmse, n_coef, N, params_per_coef=2):
    k = params_per_coef * n_coef
    return _neg2logL(rmse, N) + 2 * k


def bic(rmse, n_coef, N, params_per_coef=2):
    k = params_per_coef * n_coef
    return _neg2logL(rmse, N) + k * np.log(N)
