import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np, pandas as pd
from collections import deque

class random_decision_tree:
    def __init__(self, train_data_file_name, test_data_file_name):
        # finding folder path and converting csv to dataframe
        self.folder_path = os.path.dirname(Path(__file__).resolve())
        self.test_data_path = os.path.join(self.folder_path, test_data_file_name)
        self.train_data_path = os.path.join(self.folder_path, train_data_file_name)

        self.train_df = pd.read_csv(self.train_data_path)
        self.train_df = self.add_lags(self.train_df)
        self.x_df = self.train_df.loc[:, 'time':'N']
        self.y_df = self.train_df.loc[:, ['Y1', 'Y2']]

        lag_cols = [c for c in self.train_df.columns if ('_lag' in c or '_roll' in c)]
        self.x_df = pd.concat([self.x_df, self.train_df[lag_cols]], axis = 1)

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
        self.model = RandomForestRegressor(n_estimators=800, bootstrap=True, oob_score=True, n_jobs=-1)

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
    
    def add_lags(self, df, cols = ('Y2',), lags = (1,5,10,25,100), rolls = (5,10,25,100)):
        df = df.copy()
        for target in cols:

            past = df[target].shift(1)
            exp_mean = past.expanding(min_periods = 1).mean()
            exp_std = past.expanding(min_periods = 2).std()

            exp_mean2 = exp_mean.fillna(0.0)
            exp_std2 = exp_std.fillna(0.0)
            for l in lags:
                col = df[target].shift(l)
                df[f'{target}_lag{l}'] = col.fillna(exp_mean)
            for r in rolls:
                roll_mean = past.rolling(r, min_periods = r).mean()
                roll_std = past.rolling(r, min_periods = r).std()
                df[f'{target}_rollmean{r}'] = roll_mean.fillna(exp_mean2)
                df[f'{target}_rollstd{r}'] = roll_std.fillna(exp_std2)
        return df
    
    def predict_external_continuation(self, lags=(1,), rolls=(5,10)):
        K = max(max(lags), max(rolls))
        feature_order = list(self.x_df.columns)
        base_X = self.test_df.loc[:, 'time':'N'].reset_index(drop=True)
        y2 = self.train_df['Y2'].astype(float).values
        win = deque((y2[-K:] if len(y2) >= K else y2).tolist(), maxlen=K)
        preds = []
        for i in range(len(base_X)):
            w = np.array(win, dtype=float)
            m = w.mean() if w.size else 0.0
            s = (w.std(ddof=1) if w.size > 1 else 0.0)
            lag_feats = {f"Y2_lag{L}": (float(w[-L]) if w.size >= L else m) for L in lags}
            roll_feats = {}
            for r in rolls:
                if w.size >= r:
                    seg = w[-r:]
                    roll_feats[f"Y2_rollmean{r}"] = float(seg.mean())
                    roll_feats[f"Y2_rollstd{r}"]  = float(seg.std(ddof=1))
                else:
                    roll_feats[f"Y2_rollmean{r}"] = m
                    roll_feats[f"Y2_rollstd{r}"]  = s
            row = {c: base_X.iloc[i][c] for c in base_X.columns}
            row.update(lag_feats); row.update(roll_feats)
            for c in feature_order:
                row.setdefault(c, 0.0)
            X_row = pd.DataFrame([row], columns=feature_order)
            y_hat = self.model.predict(X_row)[0]
            preds.append(y_hat)
            win.append(float(y_hat[1]))
        return pd.DataFrame(np.asarray(preds), columns=["Y1_pred","Y2_pred"])

                

if __name__ == '__main__':
    rdt = random_decision_tree('data/train.csv', 'data/test.csv')
    rdt.testing_model_with_split_data()
    # rdt.check_df()
    rdt.initialise_train_predict_model()
    rdt.evaluate_model()
    
    rdt._ran_testing_model_with_split_data = False
    rdt.initialise_train_predict_model()
    ext_predictions = rdt.predict_external_continuation()

    into_csv = pd.DataFrame({'id': rdt.test_df['id'].values,
                            'Y1': ext_predictions['Y1_pred'],
                            'Y2': ext_predictions['Y2_pred'].values})

    into_csv.to_csv('submission.csv', index = False)
