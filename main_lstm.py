from pathlib import Path
from src.core.dataset import Dataset
from src.search.lstm_grid_search import GridSearchLSTM

def main():
    # Carrega os dados do CSV
    dataset = Dataset.from_csv("data/raw/dadosIniciais.csv")

    # Configurando caminhos para salvar os arquivos
    results_path = Path('./results')
    results_path.mkdir(exist_ok=True)
    Path('./results/models').mkdir(exist_ok=True)

    # Divide os dados em treino, validação e teste
    train, val, test = dataset.split(train=0.6, val=0.2)

    # Parametros do gridsearch
    param_grid = {
        "windows_size": [10, 20, 30],
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 2, 3, 4, 5],
        "learning_rate": [0.0001],
        "dropout": [0.0, 0.5],
        "batch_size": [128, 512, 1024],
        "epoch": [30, 40, 50]
    }

    # Definindo device (prioridade para gpu caso haja)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    grid_search_lstm(
        train_normalized,
        val_normalized,
        scaler_y,
        param_grid,
        device,
        results_path
    )

if __name__ == "__main__":
    main()