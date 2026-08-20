import json
import torch
from pathlib import Path
from src.core.dataset import Dataset
from src.search.lstm_grid_search import GridSearchLSTM

def main():
    # Carrega os dados do CSV
    dataset = Dataset.from_csv("data/raw/dadosIniciais.csv")

    # Configurando caminhos para salvar os arquivos
    results_path = Path('./results')
    results_path.mkdir(exist_ok=True)

    # Divide os dados em treino, validação e teste
    train, val, test = dataset.split(train=0.6, val=0.2)

    # Definindo device (prioridade para gpu caso haja)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parametros do gridsearch
    param_grid = {
        "windows_size": [10, 20, 30],
        "hidden_size": [64, 128, 256],
        "num_layers": [1, 2, 3, 4, 5],
        "learning_rate": [0.001],
        "dropout": [0.0, 0.5],
        "batch_size": [128, 512, 1024],
    }

    # Criando objeto GridSearchLSTM
    gsearch = GridSearchLSTM(train, val, param_grid, device)
    best_metric, best_model, best_params = gsearch.run(results_path)

    # Salvando resultados
    dict_results = {"best_evm": best_metric}
    dict_results.update(best_params)

    with open(results_path/"best_result.json", "w") as f:
        json.dump(dict_results, f)

    # Salvando modelo
    torch.save(best_model.state_dict(), results_path/"best_model.pkl")

if __name__ == "__main__":
    main()