import numpy as np
from src.metrics.metrics import RMSE, EVM

y = np.array([1+1j, 2-1j, -1+2j, 0.5+0.5j])

# 1) erro zero -> ambas as métricas zeram
assert RMSE().compute(y, y) == 0
assert EVM().compute(y, y) == 0

# 2) valor conhecido: predito = verdadeiro * (1 + 0.01)
y_hat = y * 1.01                    # erro relativo de 1% em amplitude
print("RMSE:", RMSE().compute(y, y_hat))
print("EVM :", EVM().compute(y, y_hat), "%")   # deve dar ~1.0 %