import os
import sys

import numpy as np
import pandas as pd
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

def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    n_splits = cfg["modeling"]["validation"]["n_splits"]
    
    combined_path = os.path.join(processed_path, "combined_datasets.csv")
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    
    df_encoded = prepare_features(df)
    feature_cols = get_feature_columns(df_encoded)
    
    folds = generate_walk_forward_splits(df_encoded, n_splits)
    
    all_importances = []
    for i, (train_df, test_df) in enumerate(folds):
        X_train = train_df[feature_cols]
        y_train = train_df["target_volatility"]
        
        model = XGBRegressor(
            n_estimators = 200,
            max_depth = 5,
            learning_rate = 0.05,
            random_state = 42
        )
        
        model.fit(X_train, y_train)
        
        #importance gain : how useful it is
        importance_dict = model.get_booster().get_score(importance_type="gain")
        all_importances.append(importance_dict)
    
    importance_df = pd.DataFrame(all_importances).fillna(0)
    avg_importance = (importance_df.mean().sort_values(ascending= False))
    
    print("=== Mean Feature Importance (gain, 5 fold mean) ===\n")
    for feature, score in avg_importance.head(20).items():
        bar = "#" * int(score / avg_importance.max() * 40)
        print(f"{feature:20s} {score:8.2f}  {bar}")
        
    
    #Multicollinetory control
    print(f"Correlation between technical indicators (|r|>0.8)")
    technical_cols = [c for c in feature_cols if not c.startswith("ticker_")]
    corr_matrix = df_encoded[technical_cols].corr()
    
    high_corr_pairs = []
    
    for i in range(len(technical_cols)):
        for j in range(i + 1, len(technical_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.8:
                high_corr_pairs.append((technical_cols[i], technical_cols[j], r))
    
    if high_corr_pairs:
        for f1,f2, r in sorted(high_corr_pairs, key= lambda x: abs(x[2])):
            print(f"  {f1:20s} <-> {f2:20s}  r = {r:.3f}")
        
    else:
        print("No high correlation pairs")
    

if __name__ == "__main__":
    main()