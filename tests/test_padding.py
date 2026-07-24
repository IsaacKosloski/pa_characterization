import numpy as np
from src.core.dataset import Dataset
from src.models.memory_polynomial import MemoryPolynomial
from src.metrics.metrics import RMSE

# 1) _shift isolado — teste o helper ANTES de confiar nele
arr = np.array([1., 2., 3., 4., 5.])
s = MemoryPolynomial(memory=1)._shift
assert np.array_equal(s(arr,  2), [0., 0., 1., 2., 3.]), "atraso errado"
assert np.array_equal(s(arr, -2), [3., 4., 5., 0., 0.]), "avanço errado"
assert np.array_equal(s(arr,  0), arr)
print("_shift ok")

# 2) Phi agora tem N linhas
ds = Dataset.from_csv("../data/raw/dadosIniciais.csv")
train, val, test = ds.split()
mp = MemoryPolynomial(memory=3, r_deg=3, i_deg=3)
Phi = mp.build_regressors(train.x)
assert Phi.shape[0] == len(train), "Phi deveria ter N linhas"
assert Phi.shape[1] == (3+1)*(3+3)

# 3) predição com o mesmo comprimento do dataset
mp.fit(train)
y_pred = mp.predict(val)
assert len(y_pred) == len(val), "predição deveria ter N amostras"
print("RMSE val:", RMSE().compute(mp.align(val.y), y_pred))

# 4) comparação JUSTA entre memórias diferentes (o ganho real)
for M in [1, 3, 5]:
    m = MemoryPolynomial(memory=M, r_deg=3, i_deg=3).fit(train)
    print(f"M={M}: len={len(m.predict(val))}, RMSE={RMSE().compute(m.align(val.y), m.predict(val)):.4f}")