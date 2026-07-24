from src.core.dataset import Dataset
from src.models.generalized_memory_polynomial import GeneralizedMemoryPolynomial
from src.metrics.metrics import RMSE, EVM

ds = Dataset.from_csv("../data/raw/dadosIniciais.csv")
train, val, test = ds.split()

g = GeneralizedMemoryPolynomial(memory=2, r_deg=3, i_deg=3, lag_depth=1, lead_depth=1)
Phi = g.build_regressors(train.x)
print("Phi:", Phi.shape)
assert Phi.shape[0] == len(train)         # N linhas (zero-padding)
assert Phi.shape[1] == 42                  # a conta corrigida

g.fit(train)
print("GMP RMSE val:", RMSE().compute(g.align(val.y), g.predict(val)))
# compare com o ~0.3458 do MP: o GMP deve empatar ou ganhar