from src.core.dataset import Dataset
from src.models.memory_polynomial import MemoryPolynomial
from src.metrics.metrics import RMSE
from src.search.grid_search import GridSearch

ds = Dataset.from_csv("../data/raw/dadosIniciais.csv")
train, val, test = ds.split()

grid = {"memory": [1, 2, 3], "r_deg": [1, 2, 3], "i_deg": [1, 2, 3]}
gs = GridSearch(MemoryPolynomial, grid, RMSE())
best = gs.run(train, val)

print("melhor combinação:", best["params"])
print("RMSE na validação:", best["score"])

# nota honesta no teste, com o melhor modelo já treinado
model = best["model"]
y_pred = model.predict(test)
y_true = model.align(test.y)
print("RMSE no teste:", RMSE().compute(y_true, y_pred))