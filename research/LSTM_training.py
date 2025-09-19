import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('./data/train.csv')
#test_data = pd.read_csv('./data/test.csv')

train_load, val_load, test_load = train_data[0:48000], train_data[48000:64000], train_data[64000:80001]

x_train, y_train = train_load[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']].values, train_load[['Y1','Y2']].values
x_val, y_val = val_load[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']].values, val_load[['Y1','Y2']].values
x_test, y_test = test_load[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']].values, test_load[['Y1','Y2']].values

DataLoader(train_data, batch_size=32, shuffle = True)
