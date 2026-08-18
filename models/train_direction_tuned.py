import os
import sys

import pandas as pd 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from models.train_direction_baseline import prepare_features, get_feature_columns
from models.tuning import run_nested_walk_forward

def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    dataset_file = cfg["modeling"]["dataset_file"]
    n_outer_splits = cfg["modeling"]["validation"]["n_splits"]
    n_inner_splits = cfg["modeling"]["hyperparameter_search"]["n_inner_splits"]
    n_trials = cfg["modeling"]["hyperparameter_search"]["n_trials"]
    
    combined_path = os.path.join(processed_path, dataset_file)
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    
    df_encoded = prepare_features(df)
    feature_cols = get_feature_columns(df_encoded)
    
    print(f"Dataset : {dataset_file}, {len(feature_cols)} features")
    
    for model_family in ["xgboost", "lightgbm"]:
        print(f"\n{'=' * 50}\n{model_family.upper()} -- NESTED WALK-FORWARD TUNING\n{'=' * 50}")
        
        all_metrics, all_best_params = run_nested_walk_forward(
            df_encoded, feature_cols, "target_direction", model_family, "classification", 
            n_outer_splits, n_inner_splits, n_trials
        )
        
        metrics_df = pd.DataFrame(all_metrics)
        print(f"{model_family} -- Fold Summary")
        print(metrics_df.describe().loc[["mean", "std"]])
        
        print(f"{model_family} -- Best Parameters per fold")
        for i, params in enumerate(all_best_params, 1):
            print(f"Fold {i} : {params}")

if __name__ == "__main__":
    main()
    
            