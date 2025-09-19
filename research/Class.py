import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

class LSTMNeuralNet(nn.Module):
    def __init__(self, num_feat = 16, hidden_dim = 64, layers = 3, num_outputs = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size = num_feat, hidden_size = hidden_dim, num_layers = layers, batch_first = True)
        self.lin = nn.Linear(hidden_dim, num_outputs)
    def forward(self,x):
        _, (hidden_states, _) = self.lstm(x)
        return self.lin(hidden_states[-1])

class TimeDataSet(Dataset):
    def __init__(self, inputs, outputs, window_size = 8, horizon = 1):
        self.x = inputs
        self.y = outputs
        self.win = window_size
        self.hor = horizon

    def __len__(self):
        return len(self.x) - self.win - self.hor + 1
    
    def __getitem__(self, index):
        x_seq = self.x[index:index + self.win]
        y_seq = self.y[index + self.win: index + self.win + self.hor]
        if self.hor == 1:
            y_seq = y_seq.squeeze(0)
        return torch.tensor(x_seq, dtype = torch.float32), torch.tensor(y_seq, dtype = torch.float32)
