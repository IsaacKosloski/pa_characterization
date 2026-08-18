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
from torch.utils.data import DataLoader
from itertools import product

class TrainerLSTM():
    def __init__(self,model, criterion, lr, scaler, optimizer=None, device=torch.device('cpu')):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
        self.device = device
        self.scaler = scaler

    def fit(self,train_loader):
        """Treina o modelo LSTM por uma época."""
        self.model.train()
        metric_rmse, metric_evm = RMSE(), EVM()
        y_pred, y_true = [], []
        total_loss, total = 0.0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            y_pred.append(outputs.detach().cpu())
            y_true.append(targets.cpu())
            total += outputs.size(0)

        y_pred, y_true = np.concatenate(y_pred,axis=0), np.concatenate(y_true,axis=0)
        y_pred, y_true = self.scaler.inverse_transform(y_pred), self.scaler.inverse_transform(y_true)
        avg_loss = total_loss / total
        rmse = metric_rmse.compute(y_true, y_pred)
        evm = metric_evm.compute(y_true, y_pred)
        return avg_loss, rmse, evm

    def evaluate(self,val_loader):
        """Avalia o modelo LSTM no conjunto de validação."""
        self.model.eval()
        metric_rmse, metric_evm = RMSE(), EVM()
        y_pred, y_true = [], []
        total_loss, total = 0.0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                y_pred.append(outputs.detach().cpu())
                y_true.append(targets.cpu())
                total += outputs.size(0)

        y_pred, y_true = np.concatenate(y_pred,axis=0), np.concatenate(y_true,axis=0)
        y_pred, y_true = self.scaler.inverse_transform(y_pred), self.scaler.inverse_transform(y_true)
        avg_loss = total_loss / total
        rmse = metric_rmse.compute(y_true, y_pred)
        evm = metric_evm.compute(y_true, y_pred)
        return avg_loss, rmse, evm

class ExperimentMetricsAnalyser():
    # Armazena as métricas, melhores modelos e seus respectivos hiperparâmetros e histórico (loss de treino e validação) para a avaliação final
    def __init__(self):
        # Estrutura de chaves para salvar os hiperparâmetros
        self.RESULTS_PARAM_KEYS = RESULTS_PARAM_KEYS = ("ws", "hs", "nl", "lr", "dr", "bs", "ep")
        self.HISTORY_VALUES_KEYS = ("train_avg_loss", "train_rmse", "train_evm", "val_avg_loss", "val_rmse", "val_evm")

        # Estrutura para salvar melhores modelos
        self.results = {
            'rmse': {
                'value': float('inf'),
                'state_model': None,
                'hyperparameters': None,
            },
            'evm': {
                'value': float('inf'),
                'state_model': None,
                'hyperparameters': None,
            },
            'avg_loss': {
                'value': float('inf'),
                'state_model': None,
                'hyperparameters': None,
            }
        }

        # Estrutura para salvar histórico de todos os modelos testados
        self.history = {}

        # Funções auxiliares para atualização dos pesos
        self.comp = lambda x,y: x < y
        self.get_params_dict = lambda p: dict(zip(RESULTS_PARAM_KEYS, p, strict=True))

    def update_results(self, rmse, evm, avg_loss, model, params):
        for k, metric in zip(self.results.keys(),(rmse, evm, avg_loss)):
            if self.comp(self.results[k]['value'],metric):
                self.results[k]['value'] = metric
                self.results[k]['state_model'] = model.state_dict()
                self.results[k]['hyperparameters'] = self.get_params_dict(params)

    def update_history(self, key_hyperparams, values):
        # Atualiza o histórico das métricas de treino para cada modelo na ordem (train_avg_loss, train_rmse, train_evm, val_avg_loss, val_rmse, val_evm)
        if key_hyperparams[:-1] not in self.history:
            history_element = {
                f"{key_hyperparams[-1]}": {
                    "train_avg_loss": [values[0]],
                    "train_rmse": [values[1]],
                    "train_evm": [values[2]],
                    "val_avg_loss": [values[3]],
                    "val_rmse": [values[4]],
                    "val_evm": [values[5]]
                }
            }
            self.history[f"{key_hyperparams[:-1]}"] = history_element
        else:
            key = f"{key_hyperparams[:-1]}"
            n_epochs = key_hyperparams[-1]
            for metrics,value in zip(self.HISTORY_VALUES_KEYS,values):
                self.history[key][n_epochs][metrics].append(value)

    def save(self, save_path):
        # Salva os resultados dos melhores modelos e o histórico das métricas de treino e validação
        with open(f"{save_path}/best_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        with open(f"{save_path}/history.json", "w") as f:
            json.dump(self.history, f, indent=2)

class GridSearchLSTM():
    def __init__(self, train, val, test, param_grid):
        self.train = train
        self.val = val
        self.test = test
        self.param_grid = param_grid
        self.train_norm, self.val_norm, self.test_norm, self.scaler_y = self.normalize_datasets(train, val, test)
        self.scaler = None

    def get_param_combinations(self):
        return (
            list(
                product(
                    self.param_grid["windows_size"],
                    self.param_grid["hidden_size"],
                    self.param_grid["num_layers"],
                    self.param_grid["learning_rate"],
                    self.param_grid["dropout"],
                    self.param_grid["batch_size"],
                    self.param_grid["epoch"]
                )
            )
        )

    def run(self, save_path):
        # Executa o grid search sobre os hiperparâmetros e salva os resultados dos melhores modelos em arquivos
        params_combination = self.get_param_combinations()
        n_combinations = len(params_combination)

        for i, (ws, hs, nl, lr, dr, bs, n_epochs) in enumerate(params_combination):
            # Cria o modelo LSTM com os hiperparâmetros atuais
            model = PaLSTM(input_size=4, hidden_size=hs, num_layers=nl, dropout=dr).to(self.device)
            model.to(torch.float64)

            # Configurando Sliding Window Dataset para treino e validação
            slw_train = SlidingWindowDataset(self.train_norm, ws)
            slw_val = SlidingWindowDataset(self.val_norm, ws)

            # Gerando datalaoders
            train_loader = DataLoader(slw_train, batch_size=bs, shuffle=True)
            val_loader = DataLoader(slw_val, batch_size=bs, shuffle=False)

            # Configurando classe para treinar lstm
            trainer = TrainerLSTM(model,nn.MSELoss,lr,self.scaler_y)

            # Configurando classe para análise dos resultados
            result_analyser = ExperimentMetricsAnalyser()

            print(f"Combination ({i+1}/{n_combinations}): WS = {ws}, HS = {hs}, NL = {nl}, LR = {lr}, DR = {dr}, BS = {bs}, EPOCHS = {n_epochs}\n")
            for epoch in range(n_epochs):
                hyperparams = (ws, hs, nl, lr, dr, bs, n_epochs)
                train_avg_loss, train_rmse, train_evm = trainer.fit(train_loader)
                val_avg_loss, val_rmse, val_evm = trainer.evaluate(val_loader)

                if (epoch+1)%(n_epochs//10) == 0:
                    print(f"\t({epoch+1}/{n_epochs}):")
                    print(f"\t\ttrain | rmse: {train_rmse:.6f}, evm: {train_evm:.6f}, avg_loss: {train_avg_loss:.6f}")
                    print(f"\t\tval   | rmse: {val_rmse:.6f}, evm: {val_evm:.6f}, avg_loss: {val_avg_loss:.6f}\n")

                result_analyser.update_results(val_rmse, val_evm, val_avg_loss, model, hyperparams)
                result_analyser.update_history(hyperparams, (train_avg_loss, train_rmse, train_evm, val_avg_loss, val_rmse, val_evm))

        result_analyser.save(save_path)