import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Class import TimeDataSet, LSTMNeuralNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

train_data = pd.read_csv(r'C:\Users\Ardil\Documents\QuantChallenge\Coding-challenge\research\data\train.csv')
#test_data = pd.read_csv('./data/test.csv')

train_load, val_load, test_load = train_data[0:48000], train_data[48000:64000], train_data[64000:80001]

x_train, y_train = train_load[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']].values, train_load[['Y1','Y2']].values
x_val, y_val = val_load[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']].values, val_load[['Y1','Y2']].values
x_test, y_test = test_load[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']].values, test_load[['Y1','Y2']].values

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

Train_dataset = TimeDataSet(x_train, y_train)
Val_dataset = TimeDataSet(x_val, y_val)
Test_dataset = TimeDataSet(x_test, y_test)

train = DataLoader(Train_dataset, 512, True)
val = DataLoader(Val_dataset, 512, True)
test = DataLoader(Test_dataset, 512)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMNeuralNet().to(device)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.0003)
epoch_count = 100

for i in range(epoch_count):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimiser.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train)

    val_loss = 0.0
    model.eval() 
    with torch.no_grad(): 
        for x_val_batch, y_val_batch in val:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)

            val_output = model(x_val_batch) 
            loss = criterion(val_output, y_val_batch) 

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val)

    preds = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch in test:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)

            preds.append(output.cpu())
            targets.append(y_batch.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    r2_y2 = r2_score(targets[:, 1], preds[:, 1])

    print(f"R² for Y2: {r2_y2:.4f}")
    print('Epoch:' + str(i) + ": " + str(avg_train_loss) + " and val is" + str(avg_val_loss))