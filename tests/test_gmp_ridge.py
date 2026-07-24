from src.core.dataset import Dataset
from src.models.memory_polynomial import MemoryPolynomial
from src.models.generalized_memory_polynomial import GeneralizedMemoryPolynomial
from src.metrics.metrics import RMSE, EVM
from src.core.estimators import Ridge, OLS
from src.search.grid_search import GridSearch
import numpy as np
from functools import partial

ds = Dataset.from_csv("../data/raw/dadosIniciais.csv")
train, val, test = ds.split()

g = GeneralizedMemoryPolynomial(memory=2, r_deg=3, i_deg=3, lag_depth=1, lead_depth=1)
Phi = g.build_regressors(train.x)
print("Phi:", Phi.shape)
assert Phi.shape[0] == len(train)         # N linhas (zero-padding)
assert Phi.shape[1] == 42                  # a conta corrigida
print("condição de Phi:", np.linalg.cond(Phi))   # espere algo enorme: 1e8, 1e10+

g.fit(train)
print("RMSE treino:", g.evaluate(train,RMSE()))
print("RMSE val   :",  g.evaluate(val,RMSE()))

g2 = GeneralizedMemoryPolynomial(memory=2, r_deg=3, i_deg=3, lag_depth=1, lead_depth=1,
         estimator_factory=Ridge).fit(train)
print("RMSE val (Ridge):", RMSE().compute(g2.align(val.y), g2.predict(val)))

g0 = GeneralizedMemoryPolynomial(memory=2, r_deg=3, i_deg=3, lag_depth=0, lead_depth=0).fit(train)
mp = MemoryPolynomial(memory=2, r_deg=3, i_deg=3).fit(train)

print("GMP(0,0) cols:", g0.build_regressors(train.x).shape[1])  # deve ser 18
print("GMP(0,0) RMSE val:",  g0.evaluate(val,RMSE()), g0.predict(val)))
print("MP       RMSE val:",  mp.evaluate(val,RMSE()), mp.predict(val)))

grid = {
    "memory":     [1, 2, 3],
    "r_deg":      [3],
    "i_deg":      [3],
    "lag_depth":  [1, 2],
    "lead_depth": [1, 2],
    "estimator_factory": [partial(Ridge, lam=l) for l in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]],
}
best = GridSearch(GeneralizedMemoryPolynomial, grid, RMSE()).run(train, val)
print("GMP melhor:", best["params"], "RMSE val:", best["score"])

mp_best  = MemoryPolynomial(memory=3, r_deg=3, i_deg=3).fit(train)
gmp_best = GeneralizedMemoryPolynomial(memory=1, r_deg=3, i_deg=3, lag_depth=2, lead_depth=2,
               estimator_factory=partial(Ridge, lam=0.001)).fit(train)

for nome, mdl in [("MP", mp_best), ("GMP", gmp_best)]:
    yt, yp = mdl.align(test.y), mdl.predict(test)
    print(f"{nome}: RMSE={RMSE().compute(yt,yp):.4f}  EVM={EVM().compute(yt,yp):.3f}%")

yt = test.y[3:]
yp = mp_best.predict(test)[3:]
print("MP teste sem as 3 linhas de padding:", RMSE().compute(yt, yp))