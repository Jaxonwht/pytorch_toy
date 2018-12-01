import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class EmailDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.file = [(i, j) for i, j in enumerate(open(file_path))]

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        return self.file[index][1]


