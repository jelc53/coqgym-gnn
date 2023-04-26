import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class CoqGymGNNDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return torch.load(self.paths[idx])
