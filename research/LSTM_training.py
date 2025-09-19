import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')