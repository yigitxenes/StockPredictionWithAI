import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config

##Have to choose the splits from dates not rows lo prevent data leakage
def generate_walk_forward_splits(df, n_splits, date_col : str = "Date"):
    unique_dates = np.sort(df[date_col].unique())
    n_dates = len(unique_dates)
    
    split_points = np.linspace(0, n_dates, n_splits + 2, dtype=int)[1:-1]
    
    folds = []
    for i, split_idx in enumerate(split_points):
        train_end_date = unique_dates[split_idx - 1]
        
        if i+1 < len(split_points):
            test_end_date = unique_dates[split_points[i+1] -1]
        else:
            test_end_date = unique_dates[-1]
        
        train_df = df[df[date_col] <= train_end_date]
        test_df = df[(df[date_col] > train_end_date) & (df[date_col] <= test_end_date)]
        
        if len(test_df) == 0:
            continue
        
        folds.append((train_df, test_df))
    
    return folds

def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    n_splits = cfg["modeling"]["validation"]["n_splits"]    
    
    combined_path = os.path.join(processed_path, "combined_datasets.csv")
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    
    folds = generate_walk_forward_splits(df, n_splits= n_splits)
    
    print(f"Total number of folds : {len(folds)}")
    
    for i, (train_df, test_df) in enumerate(folds):
        print(f"Fold {i + 1} : ")
        print(f"  Train: {train_df['Date'].min().date()} -> {train_df['Date'].max().date()}")
        print(f"{len(train_df)} rows")
        print(f"  Test:  {test_df['Date'].min().date()} -> {test_df['Date'].max().date()}")
        print(f"{len(test_df)} rows")

        assert train_df["Date"].max() < test_df["Date"].min(), \
            f"There is leakage on fold {i + 1}"
        print("No leakage found")
    

if __name__ == "__main__":
    main()