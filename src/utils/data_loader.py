import pandas as pd
import torch
from torch.utils.data import Dataset


class PNPDataset(Dataset):
    def __init__(self, data_file: str):
        self.data = pd.read_csv(data_file)
        self.input_data = self.data.iloc[:, :-1].values
        self.output_data = self.data.iloc[:, -1:].values
        self.input_dim = self.input_data.shape[1]
        self.output_dim = self.output_data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.Tensor(self.input_data[idx])
        targets = torch.Tensor(self.output_data[idx])
        return inputs, targets

