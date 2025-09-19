import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear

class TimeDataSet(Dataset):
    def __init__(self, inputs, outputs, window_size = 8, horizon = 1):
        self.x = inputs
        self.y = outputs
        self.win = window_size
        self.hor = horizon

    def __len__(self):
        return len(self.x - self.win - self.hor + 1)
    
    def __getitem__(self, index):
        x_seq = self.x[index:index + self.win]
        y_seq = self.y[index + self.win: index + self.win + self.hor]
        return torch.tensor(x_seq, dtype = torch.float32), torch.tensor(y_seq, dtype = torch.float32)
