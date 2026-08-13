import itertools
import random

class GridSearch():

    def __init__(self, model_cls, param_grid, metric):
        self.model_cls = model_cls
        self.param_grid = param_grid
        self.metric = metric


    def run(self, train_ds, val_ds):
        self.history, self.trials = [], []

        for params in self._generate_combinations():

            assay = self._evaluate_combination(params, train_ds, val_ds)
            self.trials.append(assay)

            row = {key: (value if isinstance(value, (int, float, str, bool)) else str(value))for key, value in params.items()}
            row["score"] = assay["score"]
            row["n_coef"] = assay["n_coef"]
            row["N"] = assay["N"]
            self.history.append(row)

        best_assay = min(self.trials, key=lambda assay: assay["score"])
        return {
            "score": best_assay["score"],
            "params": best_assay["params"],
            "model": best_assay["model"],
        }

    def best_by(self, criterion):
        return min(self.trials, key=lambda t: criterion(t["score"], t["n_coef"], t["N"]))

    def _generate_combinations(self):
        keys = list(self.param_grid.keys())
        return [dict(zip(keys, values))for values in itertools.product(*self.param_grid.values())]

    def _evaluate_combination(self, params, train_ds, val_ds):
        model = self.model_cls(**params).fit(train_ds)
        return {
            "score": model.evaluate(val_ds, self.metric),
            "n_coef": len(model.coefficients()["real"]),
            "N": len(model.align(val_ds.y)),
            "params": params,
            "model": model,
        }

class RandomizedGridSearch(GridSearch):
    def __init__(self, model_cls, param_grid, metric, n_iter=100, seed=None):
        super().__init__(model_cls, param_grid, metric)
        self.n_iter = n_iter
        self.seed = seed

    def _generate_combinations(self):
        generator = random.Random(self.seed)
        keys = list(self.param_grid.keys())
        return [{key: generator.choice(self.param_grid[key]) for key in keys} for _ in range(self.n_iter)]