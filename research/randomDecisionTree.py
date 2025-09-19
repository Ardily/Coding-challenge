import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

class random_decision_tree:


    def __init__(self, file_name):
        # finding folder path and converting csv to dataframe
        self.folder_path = os.path.dirname(Path(__file__).resolve())
        self.data_path = os.path.join(self.folder_path, file_name)

        self.df = pd.read_csv(self.data_path)
        self.feature_columns = self.df.loc[:, 'time':'N']
        self.target_columns = self.df.loc[:, ['Y1', 'Y2']]

    def check_df(self):
        # visualising dataframes
        print(self.feature_columns)
        print(self.target_columns)

        print(self.feature_columns.shape)
        print(self.target_columns.shape)

    def initialise_train_model(self):
        # initialising & training model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.feature_columns, self.target_columns)

    def predict(self, test_file_name):
        # using trained model for predictions
        self.test_data_path = os.path.join(self.folder_path, test_file_name)
        self.test_data_df = pd.read_csv(self.test_data_path)
        self.test_data_df = self.test_data_df.drop('id', axis=1)

        self.predict_data = self.model.predict(self.test_data_df)


