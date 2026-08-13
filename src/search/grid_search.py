import os
import itertools
import numpy as np


for variavel in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variavel, "1")

class GridSearch():

    def __init__(self, model_cls, param_grid, metric):
        self.model_cls = model_cls
        self.param_grid = param_grid
        self.metric = metric


    def run(self, train_ds, val_ds):
        self.history, self.trials = [], []
        keys = list(self.param_grid.keys())
        best = {
            "score"  : float("inf"),
            "params" : [],
            "model"  : None
        }
        for combo in itertools.product(*self.param_grid.values()):
            params = dict(zip(keys, combo))
            model = self.model_cls(**params).fit(train_ds)
            score = model.evaluate(val_ds, self.metric)
            if score < best["score"]:
                best = {
                    "score": score,
                    "params": params,
                    "model": model
                }
            params_ok = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                         for k, v in params.items()}
            row = {
                **params_ok,
                "score": score,
                "n_coef": len(model.coefficients()["real"])
            }
            self.history.append(row)
            self.trials.append({
                "score": score,
                "n_coef": row["n_coef"],
                "N": len(model.align(val_ds.y)),
                "params": params,
                "model": model,
            })
        return best

    def best_by(self, criterion):
        return min(self.trials, key=lambda t: criterion(t["score"], t["n_coef"], t["N"]))

