import torch
from torch.utils.data import Dataset


class Spec(Dataset):
    def __init__(self, data):
        print(data.mean(), data.std())
        self.data = torch.from_numpy((data - data.mean()) / data.std())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
