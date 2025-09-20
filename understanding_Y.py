import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

train_data = pd.read_csv(r'C:\Users\Ardil\Documents\QuantChallenge\Coding-challenge\research\data\train.csv')
Y1 = train_data['Y1']
Y2 = train_data['Y2']
time = train_data['time']
plot_acf(Y2, lags=100)
plt.figure(figsize=(12, 4))
#plt.scatter(time, Y1)
#plt.scatter(time, Y2)
plt.show()
