from src.search.parallel_grid_search import ParallelGridSearch
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
gs = ParallelGridSearch(MemoryPolynomial, {"memory": [1, 2, 3], "r_deg": [3], "i_deg": [3]},
                        RMSE(), n_jobs=-1)
melhor = gs.run(treino, validacao)
print("melhor:", melhor["params"], "RMSE:", melhor["score"])
print("nº de ensaios:", len(gs.trials))
print("history[0] tem N?", "N" in gs.history[0])