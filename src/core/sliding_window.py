import torch
from torch.utils.data import Dataset

class SlidingWindowPaDataset(Dataset):
    def __init__(self, data, window_size, target_dim=1):
        self.data = data
        self.data_size = len(data["Xreal"])
        self.window_size = window_size
        self.target_dim = target_dim

    def __len__(self):
        return (self.data_size - self.window_size - self.target_dim + 1)

    def __getitem__(self, index):
        start_index = index
        end_index = start_index + self.window_size

        x = torch.stack([
            self.data["Xreal"][start_index:end_index],
            self.data["Ximag"][start_index:end_index],
            self.data["Yreal"][start_index:end_index],
            self.data["Yimag"][start_index:end_index]
        ], dim=1)

        y = torch.stack([
            self.data["Yreal"][end_index:end_index+self.target_dim],
            self.data["Yimag"][end_index:end_index+self.target_dim]
        ], dim=1).squeeze(0)

        return x,y