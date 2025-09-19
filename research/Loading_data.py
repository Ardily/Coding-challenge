import pandas as pd
import numpy as np

df = pd.read_csv('./data/train.csv')
Features = df.loc[:, 'A', 'N']
Outcomes = df[['Y1,Y2']]

def create_sequence(Features, Outcomes, window_size = 50):
    x_sequence, y_sequence = [], []
    for i in range