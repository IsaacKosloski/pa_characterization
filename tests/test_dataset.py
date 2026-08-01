from src.core.dataset import Dataset

ds = Dataset.from_csv("data/raw/dadosIniciais.csv")
print(len(ds))                            # ~48384
print(ds.x[0], ds.y[0])                   # dois números complexos
ds_comp = ds.get_components()             # analisa as components
print(ds_comp.keys())
tr, va, te = ds.split()
print(len(tr), len(va), len(te))          # ~29030 / 9677 / 9677, somando o total