import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.metrics.metrics import RMSE, EVM

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