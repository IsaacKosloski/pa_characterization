import os
import itertools
import random
from joblib import Parallel, delayed

for variavel in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variavel, "1")

class ParallelGridSearch():

    def __init__(self, model_cls, param_grid, metric, n_jobs=-1):
        self.model_cls = model_cls
        self.param_grid = param_grid
        self.metric = metric
        self.n_jobs = n_jobs

    def evaluate_combination(self, params, train_ds, val_ds):
        model = self.model_cls(**params).fit(train_ds)
        score = model.evaluate(val_ds, self.metric)
        assay = {
            "score": score,
            "n_coef": len(model.coefficients()["real"]),
            "N": len(model.align(val_ds.y)),
            "params": params,
            "model": model
        }
        return assay

    def run(self, train_ds, val_ds):
        self.history, self.trials = [], []
        combinations = self._generate_combinations()
        assays = Parallel(n_jobs=self.n_jobs)(delayed(self.evaluate_combination)(params, train_ds, val_ds) for params in combinations)
        self.trials = assays
        best_assay = min(assays, key=lambda assay: assay["score"])
        best = {
            "score": best_assay["score"],
            "params": best_assay["params"],
            "model": best_assay["model"]
        }
        for assay in assays:
            row = {chave: (valor if isinstance(valor, (int, float, str, bool)) else str(valor))
                   for chave, valor in assay["params"].items()}
            row["score"] = assay["score"]
            row["n_coef"] = assay["n_coef"]
            row["N"] = assay["N"]
            self.history.append(row)

        return best

    def best_by(self, criterion):
        return min(self.trials, key=lambda t: criterion(t["score"], t["n_coef"], t["N"]))

    def _generate_combinations(self):
        keys = list(self.param_grid.keys())
        return [dict(zip(keys, values)) for values in itertools.product(*self.param_grid.values())]


class RandomizedGridSearch(ParallelGridSearch):

    def __init__(self, model_cls, param_grid, metric, n_iter=100, n_jobs=-1, seed=None):
        super().__init__(model_cls, param_grid, metric, n_jobs)
        self.n_iter = n_iter
        self.seed = seed

    def _generate_combinations(self):
        gerador = random.Random(self.seed)
        keys = list(self.param_grid.keys())
        return [{chave: gerador.choice(self.param_grid[chave]) for chave in keys}for _ in range(self.n_iter)]