import os
import sys

import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from models.walk_forward import generate_walk_forward_splits


INDICATOR_FAMILIES = {
    "RSI": ["RSI_14"],
    "MACD": ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
    "Bollinger": ["BBL_20_2.0_2.0", "BBM_20_2.0_2.0", "BBU_20_2.0_2.0", "BBB_20_2.0_2.0", "BBP_20_2.0_2.0"],
    "ATR": ["ATRr_14"],
    "ADX": ["ADX_14", "ADXR_14_2", "DMP_14", "DMN_14"],
    "OBV": ["OBV"],
}


ALWAYS_KEEP_PATTERNS = ["Volume", "ticker"]

def prepare_features(df):
    df = df.copy()
    df = pd.get_dummies(df, columns= ["ticker"], prefix= "ticker")
    return df

def get_always_keep_cols(df):
    return [c for c in df.columns if any(c.startswith(p) or c == p for p in ALWAYS_KEEP_PATTERNS)]

def compute_family_importance(df, feature_cols, target_cols, task, n_splits):
    folds = generate_walk_forward_splits(df, n_splits)
    family_scores = {family: [] for family in INDICATOR_FAMILIES}
    
    for train_df, test_df in folds:
        X_train, y_train = train_df[feature_cols], train_df[target_cols]
        
        if task == "classification":
            model = XGBClassifier(n_estimators = 200, max_depth = 5, learning_rate = 0.05,
                                  eval_metric = "logloss", random_state = 42)
        
        else:
            model = XGBRegressor(n_estimators = 200, max_depth = 5, learning_rate = 0.05, random_state = 42)
        
        model.fit(X_train, y_train)
        gain = model.get_booster().get_score(importance_type= "gain")
        
        
        for family, cols in INDICATOR_FAMILIES.items():
            family_total = sum(gain.get(c, 0.0) for c in cols)
            family_scores[family].append(family_total)
            
    avg_scores = {family : np.mean(scores) for family, scores in family_scores.items()}
    return avg_scores


def evalute_feature_set(df, feature_cols, target_col, task, n_splits):
    folds = generate_walk_forward_splits(df, n_splits)
    fold_metrics = []
    
    for train_df, test_df in folds:
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]
        
        if task == "classification":
            model = XGBClassifier(n_estimators = 200, max_depth = 5, learning_rate = 0.05,
                                              eval_metric = "logloss", random_state = 42)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            fold_metrics.append(roc_auc_score(y_test, proba))
        
        else:
            model = XGBRegressor(n_estimators = 200, max_depth = 5, learning_rate = 0.05, random_state = 42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fold_metrics.append(np.sqrt(mean_squared_error(y_test, pred)))
            
    return np.mean(fold_metrics), np.std(fold_metrics)


def run_ablation(df, target_col, task, n_splits):
    df_encoded = prepare_features(df)
    always_keep = get_always_keep_cols(df_encoded)
    
    all_family_cols = [c for cols in INDICATOR_FAMILIES.values() for c in cols]
    full_feature_cols = always_keep + all_family_cols
    
    print(f"Calculating Family importance ({task}....)")
    importance = compute_family_importance(df_encoded, full_feature_cols, target_col, task, n_splits)
    ranked_families = sorted(importance, key=importance.get)
    print("Importance Ranks (Lower to Higher) : ")
    
    for family in ranked_families:
        print(f"{family} : {importance[family]:.2f}")
    
    results = []
    remaining_families = list(INDICATOR_FAMILIES.keys())
    
    metric_name = "auc" if task == "classification" else "rmse"
    
    while remaining_families:
        active_cols = always_keep + [c for f in remaining_families for c in INDICATOR_FAMILIES[f]]
        mean_metric, std_metric = evalute_feature_set(df_encoded, active_cols, target_col, task, n_splits)
        
        results.append({
            "n_families" : len(remaining_families),
            "families" : ", ".join(remaining_families),
            f"{metric_name}_mean" : mean_metric,
            f"{metric_name}_std" : std_metric
            
        })
        
        print(f"{len(remaining_families)} family ({", ".join(remaining_families)}):"
              f"{metric_name} = {mean_metric:.5f} (+/-{std_metric:.5f})")
        
        if not remaining_families:
            break
        
        for family in ranked_families:
            if family in remaining_families:
                remaining_families.remove(family)
                break
    
    return pd.DataFrame(results)


def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    n_splits = cfg["modeling"]["validation"]["n_splits"]
    
    combined_path = os.path.join(processed_path, "combined_datasets.csv")
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    
    print("=" * 60)
    print("DIRECTION ABLATION (AUC, higher better)")
    print("=" * 60)
    direction_results = run_ablation(df, "target_direction", "classification", n_splits)
    print("\n Summary Table:")
    print(direction_results.to_string(index=False))

    print("\n" + "=" * 60)
    print("VOLATILITY ABLATION (RMSE, lower better)")
    print("=" * 60)
    volatility_results = run_ablation(df, "target_volatility", "regression", n_splits)
    print("\nSummary Table:")
    print(volatility_results.to_string(index=False))
    

if __name__ == "__main__":
    main()
    