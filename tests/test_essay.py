from src.core.dataset import Dataset
from src.models.memory_polynomial import MemoryPolynomial
from src.search.grid_search import GridSearch
from src.metrics.metrics import RMSE

dados = Dataset.from_csv("../data/raw/dadosIniciais.csv")
treino, validacao, teste = dados.split()
max_mem = 3
max_deg = 3
grid = ("MemoryPolynomial", MemoryPolynomial, {
            "memory": list(range(1, max_mem + 1)),
            "r_deg": list(range(1, max_deg + 1)),
            "i_deg": list(range(1, max_deg + 1)),
        })
gd = GridSearch(MemoryPolynomial, grid, RMSE())
ensaio = gd.evaluate_combination({"memory": 3, "r_deg": 3, "i_deg": 3}, treino, validacao)
print(ensaio["score"], ensaio["n_coef"], ensaio["N"])
print("chaves:", list(ensaio.keys()))