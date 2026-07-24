from functools import partial
from src.core.estimators import Ridge
from src.search.grid_search import GridSearch
from src.models.generalized_memory_polynomial import GeneralizedMemoryPolynomial
from src.metrics.metrics import RMSE

grid = {
    "memory":     [1, 2, 3],
    "r_deg":      [3],
    "i_deg":      [3],
    "lag_depth":  [1, 2],
    "lead_depth": [1, 2],
    "estimator_factory": [partial(Ridge, lam=l) for l in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]],
}
best = GridSearch(GeneralizedMemoryPolynomial, grid, RMSE()).run(train, val)
print("GMP melhor:", best["params"], "RMSE val:", best["score"])