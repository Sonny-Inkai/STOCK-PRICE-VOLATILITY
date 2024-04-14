import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, x_set, y_set):
        super(Dataset).__init__()
        self.x_set = x_set
        self.y_set = y_set

    def __len__(self):
        return len(self.x_set)
    
    def __getitem__(self, idx):
        return self.x_set[idx], self.y_set[idx]