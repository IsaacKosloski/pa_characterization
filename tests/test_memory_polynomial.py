from src.core.dataset import Dataset
from src.models.memory_polynomial import MemoryPolynomial
import numpy as np

ds = Dataset.from_csv("data/raw/dadosIniciais.csv")
train, val, test = ds.split()

# Montando Phi (regressores)
m = MemoryPolynomial(memory=2, r_deg=3, i_deg=3)
Phi = m.build_regressors(train.x)
print("Phi shape: ", Phi.shape)
assert Phi.shape[1] == (2+1)*(3+3)  # 18 colunas
assert Phi.shape[0] == len(train.x) - 2 # N-M linhas

# Fit, Predict e RMSE
def rmse(y_true, y_pred): return np.sqrt(np.mean(np.abs(y_true - y_pred) ** 2))

m.fit(train)
y_pred = m.predict(val)
y_true = m.align(val.y)
print("RMSE (M=2, r_deg=3, i_deg=3): ", rmse(y_true, y_pred))

# Teste de Sanidade (o modelo linear tem maior RMSE)
lin = MemoryPolynomial(memory=0, r_deg=1, i_deg=1).fit(train)
print("RMSE linear (M=0,r=1,i=1):", rmse(lin.align(val.y), lin.predict(val)))


