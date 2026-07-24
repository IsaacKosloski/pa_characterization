from functools import partial

from src.core.dataset import Dataset
from src.core.estimators import Ridge
from src.metrics.metrics import RMSE, EVM
from src.search.grid_search import GridSearch
from src.experiments.experiment_run import ExperimentRun
from src.engine.simulator import Simulator
from src.viz.plots import Plotter

from src.models.memory_polynomial import MemoryPolynomial
from src.models.generalized_memory_polynomial import GeneralizedMemoryPolynomial
from src.models.volterra import Volterra
from src.search.selection import aic, bic

SELECTOR = bic

def build_models_grid(max_mem=8, max_deg=4, max_depth=3, min_lam_exp=-6, max_lam_exp=-2):
    LAMS = [partial(Ridge, lam=10 ** i) for i in range(min_lam_exp, max_lam_exp + 1)]

    MODELS = [
        ("MemoryPolynomial", MemoryPolynomial, {
            "memory": list(range(1, max_mem + 1)),
            "r_deg": list(range(1, max_deg + 1)),
            "i_deg": list(range(1, max_deg + 1)),
        }),
        ("GMP", GeneralizedMemoryPolynomial, {
            # Modelos como GMP e Volterra escalam o número de parâmetros muito rápido,
            # então podemos limitar a memória deles pela metade do máximo, se desejar.
            "memory": list(range(1, (max_mem // 2) + 1)),
            "r_deg": list(range(1, max_deg + 1)),
            "i_deg": list(range(1, max_deg + 1)),
            "lag_depth": list(range(1, max_depth + 1)),
            "lead_depth": list(range(1, max_depth + 1)),
            "estimator_factory": LAMS,
        }),
        ("Volterra", Volterra, {
            "memory": list(range(1, (max_mem // 2) + 1)),
            "order_r": list(range(1, max_deg + 1)),
            "order_i": list(range(1, max_deg + 1)),
            "estimator_factory": LAMS,
        }),
    ]

    return MODELS

def rodar_modelo(nome, cls, grid, train, val, test):
    gs = GridSearch(cls, grid, RMSE())
    gs.run(train, val)
    best = gs.best_by(SELECTOR)
    model = best["model"]

    run = ExperimentRun(model)
    metrics = {
        "rmse_val": best["score"],
        "rmse_test": model.evaluate(test, RMSE()),
        "evm_test":  model.evaluate(test, EVM()),
    }
    run.save_report(best["params"], metrics)
    run.save_coefficients()
    run.save_history(gs.history)


    Simulator(model).to_csv(run.artifact_path("simulation.csv"), x=test.x, y_true=test.y)

    # gráficos
    xa = model.align(test.x)
    yt = model.align(test.y)
    yp = model.align(model.predict(test))
    Plotter.dashboard(xa, yt, yp, title=nome, path=run.figure_path("dashboard"))
    Plotter.constellation(yt, yp, path=run.figure_path("constellation"))
    Plotter.am_am(xa, yt, yp, path=run.figure_path("am_am"))
    Plotter.am_pm(xa, yt, yp, path=run.figure_path("am_pm"))

    n = len(model.coefficients()["real"])
    print(f"{nome:18} coef={n:4d}  RMSE={metrics['rmse_test']:.4f}  "
          f"EVM={metrics['evm_test']:.3f}%  ->  {run.path}")
    return {"nome": nome, "n_coef": n, **metrics}


def main():
    ds = Dataset.from_csv("data/raw/dadosIniciais.csv")
    train, val, test = ds.split()
    print(f"amostras: treino={len(train)}  val={len(val)}  teste={len(test)}\n")

    MODELS_ = build_models_grid(max_mem=10, max_deg=15, max_depth=5)
    resultados = [rodar_modelo(n, c, g, train, val, test) for n, c, g in MODELS_]

    melhor = min(resultados, key=lambda r: r["rmse_test"])
    print(f"\nMELHOR modelo global: {melhor['nome']}  "
          f"(RMSE={melhor['rmse_test']:.4f}, EVM={melhor['evm_test']:.3f}%, "
          f"{melhor['n_coef']} coeficientes)")


if __name__ == "__main__":
    main()
