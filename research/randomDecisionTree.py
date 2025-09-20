import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class random_decision_tree:


    def __init__(self, train_data_file_name, test_data_file_name):
        # finding folder path and converting csv to dataframe
        self.folder_path = os.path.dirname(Path(__file__).resolve())
        self.test_data_path = os.path.join(self.folder_path, test_data_file_name)
        self.train_data_path = os.path.join(self.folder_path, train_data_file_name)

        self.train_df = pd.read_csv(self.train_data_path)
        self.x_df = self.train_df.loc[:, 'time':'N']
        self.y_df = self.train_df.loc[:, ['Y1', 'Y2']]

        self.test_df = pd.read_csv(self.test_data_path)
        self.x_test = self.test_df.loc[:, 'time':'N']

        self._ran_testing_model_with_split_data = False

    def testing_model_with_split_data(self):
        # splitting training data
        self.x_train = self.x_df.iloc[0:60000]
        self.y_train = self.y_df.iloc[0:60000]

        self.x_test = self.x_df.iloc[60000:70000]
        self.y_test = self.y_df.iloc[60000:70000]

        self.x_val = self.x_df.iloc[70000:80000]
        self.y_val = self.y_df.iloc[70000:80000]

        self._ran_testing_model_with_split_data = True

    def check_df(self):
        # visualising dataframes
        if self._ran_testing_model_with_split_data:
            print(self.x_train)
            print(self.y_train)

            print(self.x_test)
            print(self.y_test)

            print(self.x_val)
            print(self.y_val)

        else:
            print(self.x_df)
            print(self.y_df)

    def initialise_train_predict_model(self):
        # initialising, training model & making prediction
        self.model = RandomForestRegressor(n_estimators=800, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1)

        if self._ran_testing_model_with_split_data:
            self.model.fit(self.x_train, self.y_train)

        else:
            self.model.fit(self.x_df, self.y_df)
        
        self.y_predict = self.model.predict(self.x_test)

    def evaluate_model(self):

        if self._ran_testing_model_with_split_data:
            # calculating mean root square (mse) & R^2 score
            self.y1_mse = mean_squared_error(self.y_test['Y1'], self.y_predict[:, 0])
            self.y2_mse = mean_squared_error(self.y_test['Y2'], self.y_predict[:, 1])

            self.y1_r2 = r2_score(self.y_test['Y1'], self.y_predict[:, 0])
            self.y2_r2 = r2_score(self.y_test['Y2'], self.y_predict[:, 1])

            # the same as above but for x_val & y_val
            self.new_y_predict = self.model.predict(self.x_val)

            self.y1_val_mse = mean_squared_error(self.y_val['Y1'], self.new_y_predict[:, 0])
            self.y2_val_mse = mean_squared_error(self.y_val['Y2'], self.new_y_predict[:, 1])

            self.y1_val_r2 = r2_score(self.y_val['Y1'], self.new_y_predict[:, 0])
            self.y2_val_r2 = r2_score(self.y_val['Y2'], self.new_y_predict[:, 1])

            print('Evaluating Model with Test Data:')
            print(f'Y1 Mean Squared Error: {self.y1_mse}. Y1 R^2 Score: {self.y1_r2}')
            print(f'Y2 Mean Squared Error: {self.y2_mse}. Y2 R^2 Score: {self.y2_r2}')

            print('\n Evaluating Model with Validation Data:')
            print(f'Y1 Mean Squared Error: {self.y1_val_mse}. Y1 R^2 Score: {self.y1_val_r2}')
            print(f'Y2 Mean Squared Error: {self.y2_val_mse}. Y2 R^2 Score: {self.y2_val_r2}')

        else:
            pass


if __name__ == '__main__':
    rdt = random_decision_tree('data/train.csv', 'data/test.csv')
    rdt.testing_model_with_split_data()
    # rdt.check_df()
    rdt.initialise_train_predict_model()
    rdt.evaluate_model()



