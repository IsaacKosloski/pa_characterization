import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.models.lstm import PaLSTM
from src.core.dataset import Dataset
from src.metrics.metrics import RMSE,EVM
from src.core.sliding_window import SlidingWindowDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from itertools import product

def normalize_datasets(train, val, test=None):
    """Normaliza os datasets de treino, validação e teste usando o StandardScaler do sklearn."""
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Ajusta o scaler nos dados de treino
    train_x = scaler_x.fit_transform(np.stack([train.x.real, train.x.imag],axis=-1))
    train_y = scaler_y.fit_transform(np.stack([train.y.real, train.y.imag],axis=-1))

    # Aplica a normalização nos datasets de validação e teste
    val_x = scaler_x.transform(np.stack([val.x.real, val.x.imag],axis=-1))
    val_y = scaler_y.transform(np.stack([val.y.real, val.y.imag],axis=-1))
    test_x = scaler_x.transform(np.stack([test.x.real, test.x.imag],axis=-1))
    test_y = scaler_y.transform(np.stack([test.y.real, test.y.imag],axis=-1))

    # Cria novos objetos Dataset com os dados normalizados
    train_normalized = Dataset(train_x[:,0] + 1j * train_x[:,1], train_y[:,0] + 1j * train_y[:,1])
    val_normalized = Dataset(val_x[:,0] + 1j * val_x[:,1], val_y[:,0] + 1j * val_y[:,1])
    test_normalized = Dataset(test_x[:,0] + 1j * test_x[:,1], test_y[:,0] + 1j * test_y[:,1])

    return train_normalized, val_normalized, test_normalized, scaler_y

def fit(model, train_loader, criterion, optimizer, device, scaler_y):
    """Treina o modelo LSTM por uma época."""
    model.train()
    metric_rmse, metric_evm = RMSE(), EVM()
    y_pred, y_true = [], []
    total_loss, total = 0.0, 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        y_pred.append(outputs.detach().cpu())
        y_true.append(targets.cpu())
        total += outputs.size(0)

    y_pred, y_true = np.concatenate(y_pred,axis=0), np.concatenate(y_true,axis=0)
    y_pred, y_true = scaler_y.inverse_transform(y_pred), scaler_y.inverse_transform(y_true)
    avg_loss = total_loss / total
    rmse = metric_rmse.compute(y_true, y_pred)
    evm = metric_evm.compute(y_true, y_pred)
    return avg_loss, rmse, evm

def evaluate(model, val_loader, criterion, device, scaler):
    """Avalia o modelo LSTM no conjunto de validação."""
    model.eval()
    metric_rmse, metric_evm = RMSE(), EVM()
    y_pred, y_true = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            y_pred.append(outputs.detach().cpu())
            y_true.append(targets.cpu())
            total += outputs.size(0)

    y_pred, y_true = np.concatenate(y_pred,axis=0), np.concatenate(y_true,axis=0)
    y_pred, y_true = scaler.inverse_transform(y_pred), scaler.inverse_transform(y_true)
    avg_loss = total_loss / total
    rmse = metric_rmse.compute(y_true, y_pred)
    evm = metric_evm.compute(y_true, y_pred)
    return avg_loss, rmse, evm

def update_best_model(metric, best_metric, model, best_model, comparation):
    if comparation(metric, best_metric):
        return metric, model.state_dict(), True
    else:
        return best_metric, best_model, False

def grid_search_lstm(train, val, scaler, params, device,save_path):
    # Configurando combinações de hiperparametros do gridsearch
    WS, HS, NL, LR, DR, BS = "windows_size", "hidden_size", "num_layers","learning_rate", "dropout", "batch_size"
    params_combination = list(product(params[WS],params[HS],params[NL],params[LR],params[DR],params[BS],params["epoch"]))
    n_combinations = len(params_combination)

    # Estrutura para salvar melhores modelos
    best_metrics = {"rmse": float('inf'), "evm": float('inf'), "avg_loss": float('inf')}
    best_model = {"rmse": None, "evm": None, "avg_loss": None}
    best_params = {"rmse": None, "evm": None, "avg_loss": None}

    # Funções lambda auxiliares (comparação de métricas e atualização de melhores hiperparametros)
    comp = lambda x,y: x < y
    update_params = lambda ws, hs, nl, lr, dr, bs, ep: {"ws":ws,"hs":hs,"nl":nl,"lr":lr,"dr":dr,"bs":bs,"ep":ep}

    for i, (ws, hs, nl, lr, dr, bs, n_epochs) in enumerate(params_combination):
        # Cria o modelo LSTM com os hiperparâmetros atuais
        model = PaLSTM(input_size=4, hidden_size=hs, num_layers=nl, dropout=dr).to(device)
        model.to(torch.float64)

        # Configurando Sliding Window Dataset para treino e validação
        slw_train = SlidingWindowDataset(train, ws)
        slw_val = SlidingWindowDataset(val, ws)

        # Gerando datalaoders
        train_loader = DataLoader(slw_train, batch_size=bs, shuffle=True)
        val_loader = DataLoader(slw_val, batch_size=bs, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        print(f"Combination ({i+1}/{n_combinations}): WS = {ws}, HS = {hs}, NL = {nl}, LR = {lr}, DR = {dr}, BS = {bs}, EPOCHS = {n_epochs}\n")
        for epoch in range(n_epochs):
            train_avg_loss, train_rmse, train_evm = fit(model, train_loader, criterion, optimizer, device, scaler)
            val_avg_loss, val_rmse, val_evm = evaluate(model, val_loader, criterion, device, scaler)

            if (epoch+1)%(n_epochs//10) == 0:
                print(f"\t({epoch+1}/{n_epochs}):")
                print(f"\t\ttrain | rmse: {train_rmse:.6f}, evm: {train_evm:.6f}, avg_loss: {train_avg_loss:.6f}")
                print(f"\t\tval   | rmse: {val_rmse:.6f}, evm: {val_evm:.6f}, avg_loss: {val_avg_loss:.6f}\n")

            best_metrics["rmse"], best_model["rmse"], changed = update_best_model(
                val_rmse, best_metrics["rmse"], model, best_model["rmse"], comp
            )
            if changed:
                best_params["rmse"] = update_params(ws, hs, nl, lr, dr, bs, n_epochs)
                torch.save(best_model["rmse"], save_path/"models/rmse.pth")

            best_metrics["evm"], best_model["evm"], changed = update_best_model(
                val_evm, best_metrics["evm"], model, best_model["evm"], comp
            )
            if changed:
                best_params["evm"] = update_params(ws, hs, nl, lr, dr, bs, n_epochs)
                torch.save(best_model["evm"], save_path/"models/evm.pth")

            best_metrics["avg_loss"], best_model["avg_loss"], changed = update_best_model(
                val_avg_loss, best_metrics["avg_loss"], model, best_model["avg_loss"], comp
            )
            if changed:
                best_params["avg_loss"] = update_params(ws, hs, nl, lr, dr, bs, n_epochs)
                torch.save(best_model["avg_loss"], save_path/"models/avg_loss.pth")
            
    results = {
        "rmse": {
            "value": best_metrics["rmse"],
            "hyper_params": best_params["rmse"]
        },
        "evm": {
            "value": best_metrics["evm"],
            "hyper_params": best_params["evm"]
        },
        "avg_loss": {
            "value": best_metrics["avg_loss"],
            "hyper_params": best_params["avg_loss"]
        }
    }
    with open(f"{save_path}/results.json", "w") as f:
        json.dump(results, f, indent=2)

def main():
    # Carrega os dados do CSV
    dataset = Dataset.from_csv("data/raw/dadosIniciais.csv")

    # Configurando caminhos para salvar os arquivos
    results_path = Path('./results')
    results_path.mkdir(exist_ok=True)
    Path('./results/models').mkdir(exist_ok=True)

    # Divide os dados em treino, validação e teste
    train, val, test = dataset.split(train=0.6, val=0.2)

    # Normaliza os datasets
    train_normalized, val_normalized, test_normalized, scaler_y = normalize_datasets(train, val, test)

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