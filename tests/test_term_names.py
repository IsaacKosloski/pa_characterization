from src.core.dataset import Dataset
from src.core.estimators import Ridge
from src.metrics.metrics import RMSE, EVM
from src.search.grid_search import GridSearch
from src.viz.plots import Plotter
from src.experiments.experiment_run import ExperimentRun
from src.models.memory_polynomial import MemoryPolynomial
from src.models.generalized_memory_polynomial import GeneralizedMemoryPolynomial

ds = Dataset.from_csv("../data/raw/dadosIniciais.csv")
train, val, test = ds.split()
g = GeneralizedMemoryPolynomial(memory=2, r_deg=3, i_deg=3, lag_depth=1, lead_depth=1)
Phi = g.build_regressors(train.x)
assert Phi.shape[1] == len(g.term_names)      # nomes batem com colunas
print(g.term_names[:6])   # dá uma olhada nos primeiros termos