import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

TRAIN_CSV = r'C:\Users\Ardil\Documents\Test\train.csv'
EXTERNAL_TEST_CSV = r'C:\Users\Ardil\Documents\Test\test.csv'
OUT_PRED_CSV = 'predictions.csv'

DEVICE = 'cuda'
FEATURE_COLS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
Y_COLS = ['Y1','Y2']

Y2_LAGS = [1,2,3,5,10,20,30]
Y2_ROLLS = [5,10,20]
EXO_LAGS = [1]
EXO_EMAS = [5,10]

def add_exogenous_history(frame:pd.DataFrame):
    out = frame.copy()
    for c in FEATURE_COLS:
        for l in EXO_LAGS:
            out[f'{c}_lag{l}'] = out[c].shift(l)
        for w in EXO_EMAS:
            out[f'{c}_ema{w}'] = out[c].ewm(span=w, adjust = False).mean().shift(1)
    return out

def add_y2_memory(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for L in Y2_LAGS:
        out[f'Y2_lag{L}'] = out['Y2'].shift(L)
    for W in Y2_ROLLS:
        out[f'Y2_ema{W}']      = out['Y2'].ewm(span=W, adjust=False).mean().shift(1)
        out[f'Y2_rollmean{W}'] = out['Y2'].rolling(W).mean().shift(1)
        out[f'Y2_rollstd{W}']  = out['Y2'].rolling(W).std().shift(1)
    return out

def fe_y2(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_exogenous_history(frame)
    out = add_y2_memory(out)
    return out

def fe_y1_reg(frame: pd.DataFrame) -> pd.DataFrame:
    return add_exogenous_history(frame)

def split_with_context(prev_df, df, context_rows: int, fe_fn):
    if prev_df is None or len(prev_df) == 0:
        combo = df.copy()
        warm = 0
    else:
        tail = prev_df.tail(context_rows)
        combo = pd.concat([tail, df], axis=0)
        warm = len(tail)
    feat = fe_fn(combo).dropna()
    if warm > 0:
        feat = feat.iloc[warm:]
    return feat

def predict_best(bst, dmatrix):
    if hasattr(bst, "best_iteration") and bst.best_iteration is not None:
        return bst.predict(dmatrix, iteration_range=(0, bst.best_iteration+1))
    return bst.predict(dmatrix)

def add_exogenous_history_by_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    orig_index = out.index
    if 'time' in out.columns:
        out = out.sort_values(['id', 'time'])
    else:
        out = out.sort_values(['id'])
    for c in FEATURE_COLS:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    g = out.groupby('id', sort=False)
    for c in FEATURE_COLS:
        for L in EXO_LAGS:
            out[f'{c}_lag{L}'] = g[c].shift(L)
        for W in EXO_EMAS:
            out[f'{c}_ema{W}'] = g[c].transform(
                lambda s: s.ewm(span=W, adjust=False).mean().shift(1)
            )
    out = out.reindex(orig_index)
    return out

def tail_per_id(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if 'time' in df.columns:
        df = df.sort_values(['id', 'time'])
    else:
        df = df.sort_values(['id'])
    return df.groupby('id', sort=False).tail(n)

def predict_on_continuation(prev_df: pd.DataFrame,
                            new_df: pd.DataFrame,
                            bst_y1: xgb.Booster,
                            bst_y2: xgb.Booster,
                            context_rows: int) -> pd.DataFrame:
    print(f"[PREDICT] Using {context_rows} warm-up rows per id...")
    ctx = tail_per_id(prev_df, context_rows)
    combo = pd.concat([ctx, new_df], axis=0, ignore_index=True)
    fe_all = add_exogenous_history_by_id(combo)
    fe_new = fe_all.iloc[len(ctx):].copy().reset_index(drop=True)
    fe_new = fe_new.fillna(0)
    print(f"[PREDICT] Feature matrix rows: {len(fe_new)}")
    X = fe_new.drop(columns=['id'], errors='ignore').values
    dmat = xgb.DMatrix(X)
    print("[PREDICT] Predicting Y1...")
    y1_pred = predict_best(bst_y1, dmat)
    print("[PREDICT] Predicting Y2...")
    y2_pred = predict_best(bst_y2, dmat)
    out = fe_new[['id']].copy()
    out['Y1'] = y1_pred
    out['Y2'] = y2_pred
    print("[PREDICT] Predictions complete.")
    return out

def write_predictions_csv(pred_df: pd.DataFrame, filename: str = "predictions.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    pred_df.to_csv(path, index=False)
    print(f"Saved predictions to: {path} (rows={len(pred_df)})")
    if len(pred_df) > 0:
        print(pred_df.head())

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_CSV)
    if 'id' not in df.columns:
        df['id'] = 0  

    print("[TRAIN] Building full-dataset features for Y2...")
    context_y2 = max(max(Y2_LAGS), max(Y2_ROLLS))
    Xy2_full = split_with_context(None, df, context_y2, fe_y2)
    X_train_y2, y_train_y2 = Xy2_full.drop(columns=Y_COLS).values, Xy2_full['Y2'].values
    dtrain_y2 = xgb.DMatrix(X_train_y2, label=y_train_y2)

    params_y2 = {
        "objective":"reg:squarederror",
        "eta":0.03,
        "tree_method":"hist",
        "device":DEVICE,
        "max_depth":5,
        "min_child_weight":1,
        "subsample":0.6,
        "colsample_bytree":1.0,
        "gamma":0.0,
        "reg_lambda":1.0,
    }
    print("[Y2] Training on full data...")
    bst_y2 = xgb.train(params_y2, dtrain_y2, num_boost_round=1500)

    print("[TRAIN] Building full-dataset features for Y1...")
    context_y1 = max([1] + EXO_EMAS)
    Xy1_full = split_with_context(None, df, context_y1, fe_y1_reg)
    X_train_y1, y_train_y1 = Xy1_full.drop(columns=Y_COLS).values, Xy1_full['Y1'].values
    dtrain_y1 = xgb.DMatrix(X_train_y1, label=y_train_y1)

    params_y1 = {
        "objective":"reg:squarederror",
        "eta":0.03,
        "tree_method":"hist",
        "device":DEVICE,
        "max_depth":6,
        "min_child_weight":1,
        "subsample":0.6,
        "colsample_bytree":0.6,
        "gamma":1.0,
        "reg_lambda":1.0,
    }
    print("[Y1] Training on full data...")
    bst_y1 = xgb.train(params_y1, dtrain_y1, num_boost_round=1500)

    print("[MAIN] Loading external test (continuation)...")
    test_df = pd.read_csv(EXTERNAL_TEST_CSV)
    history_df = df[['id','time'] + FEATURE_COLS].copy()
    context_rows = max(EXO_LAGS + EXO_EMAS)  # 10
    print("[MAIN] Predicting on continuation...")
    pred_df = predict_on_continuation(history_df,
                                      test_df[['id','time'] + FEATURE_COLS].copy(),
                                      bst_y1, bst_y2,
                                      context_rows=context_rows)
    print("[MAIN] Final prediction shape:", pred_df.shape)
    write_predictions_csv(pred_df, OUT_PRED_CSV)
