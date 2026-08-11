import numpy as np
import pandas as pd
import torch

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_csv(cls, csv_path):
        """Devolve um objeto com dois sinais complexos: entrada x e saída y do PA, lidos do csv."""
        df = pd.read_csv(csv_path)
        x = df["Xreal"].to_numpy() + 1j * df["Ximg"].to_numpy()
        y = df["Yreal"].to_numpy() + 1j * df["Yimg"].to_numpy()
        return cls(x, y)

    def split(self, train=0.6, val=0.2):
        """Corta por índice, sem emabaralhar."""
        n     = len(self.x)
        i_tr  = int(n * train)  # Fim do treino
        i_val = int(n * (train + val))    # Fim da validação
        return (
            Dataset(self.x[:i_tr],      self.y[:i_tr]),          # treino    = 60% mais antigos
            Dataset(self.x[i_tr:i_val], self.y[i_tr:i_val]),     # validação = 20% seguintes
            Dataset(self.x[i_val:],     self.y[i_val:]),         # teste     = 20% finais
        )

    def __len__(self):
        return len(self.x)