from pathlib  import Path
from datetime import datetime
import pandas as pd
import numpy  as np
import json



class ExperimentRun():

    def __init__(self, model, base="output"):
        self.model = model
        self.nome = type(model).__name__
        self.pasta = Path(base) / self.nome

        existentes = [int(p.name[3:]) for p in self.pasta.glob("run*") if p.name[3:].isdigit()]
        n = max(existentes) + 1 if existentes else 1

        self.path = self.pasta / f"run{n:02d}"
        self.path.mkdir(parents=True, exist_ok=True)

    def save_report(self, params, metrics):
        params_ok = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                     for k, v in params.items()}
        saves = {
            "timestamp": datetime.now().isoformat(),
            "model": self.nome,
            "params": params_ok,
            "metrics": metrics,
            "n_coefficients": len(self.model.coefficients()["real"]),
            "term_names": self.model.term_names
        }
        with open(self.path / "report.json", "w") as f:
            json.dump(saves, f, indent=2)

    def save_coefficients(self):
        c = self.model.coefficients()
        np.savez(self.path/"coefficients.npz", real=c["real"], imag=c["imag"])

    def figure_path(self, nome):
        return self.path / f"{nome}.png"

    def save_history(self, history):
        pd.DataFrame(history).to_csv(self.path/"gridsearch.csv", index=False)

    def artifact_path(self, nome):
        return self.path / nome


