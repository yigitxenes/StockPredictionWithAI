import os
import sys

import numpy as np
import pandas as pd 

from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from models.walk_forward import generate_walk_forward_splits


NON_FEATURE_COLS = ["Date", "target_direction", "target_volatility", "Unnamed: 0",
                    "Open", "High", "Low", "Close"]

def prepare_features(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=["ticker"], prefix="ticker")
    return df

def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]

def naive_baseline_predict(train_df, test_df):
    """Naive baseline: Uses the LAST known volatility value
    in the train set for each ticker as the prediction for all 
    of that ticker's rows in the test set.
    Goal: To see if the model actually outperforms this simple strategy."""
    predictions = np.zeros(len(test_df))
    test_df = test_df.reset_index(drop = True)
    
    ticker_cols = [c for c in train_df.columns if c.startswith("ticker_")]
    
    for idx,row in test_df.iterrows():
        active_ticker_col = [c for c in ticker_cols if row[c] == 1]
        
        if not active_ticker_col:
            predictions[idx] = train_df["target_volatility"].mean()
            continue
        
        ticker_col = active_ticker_col[0]
        ticker_train_rows = train_df[train_df[ticker_col] == 1]
        last_known_vol = ticker_train_rows["target_volatility"].iloc[-1]
        predictions[idx] = last_known_vol
    
    return predictions


def train_and_evaluate_folds(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols]
    y_train = train_df["target_volatility"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_volatility"]
    
    
    model = XGBRegressor(
        n_estimators = 200,
        max_depth = 5,
        learning_rate = 0.05,
        random_state = 42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    naive_pred = naive_baseline_predict(train_df, test_df)
    
    metrics = {
        "rmse_model" : np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae_model" : mean_absolute_error(y_test, y_pred),
        "rmse_naive" : np.sqrt(mean_squared_error(y_test, naive_pred)),
        "mae_naive" : mean_absolute_error(y_test, naive_pred)
    }
    
    return metrics

def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    n_splits = cfg["modeling"]["validation"]["n_splits"]
    
    combined_path = os.path.join(processed_path, "combined_datasets_with_sentiments.csv")
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    
    df_encoded = prepare_features(df)
    feature_cols = get_feature_cols(df_encoded)
    
    folds = generate_walk_forward_splits(df_encoded, n_splits)
    all_metrics = []
    
    for i,(train_df, test_df) in enumerate(folds):
        metrics = train_and_evaluate_folds(train_df,test_df, feature_cols)
        all_metrics.append(metrics)
        
        print(f"Fold : {i+1}")
        print(f"  Model  -> RMSE: {metrics['rmse_model']:.5f}  MAE: {metrics['mae_model']:.5f}")
        print(f"  Naive  -> RMSE: {metrics['rmse_naive']:.5f}  MAE: {metrics['mae_naive']:.5f}")
        improvement = (1 - metrics['rmse_model'] / metrics['rmse_naive']) * 100
        print(f"Model has improved RMSE %{improvement} than naive")
        
    avg_metrics = pd.DataFrame(all_metrics)
    print("Summary Between Folds")
    print(avg_metrics.describe().loc[["mean", "std"]])
    

if __name__ == "__main__":
    main()
    
    