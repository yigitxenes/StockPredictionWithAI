import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

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

def train_and_evalute_folds(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols]
    y_train = train_df["target_direction"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["target_direction"]
    
    model = XGBClassifier(
        n_estimators = 200,
        max_depth = 5,
        learning_rate = 0.05,
        eval_metric = "logloss",
        random_state = 42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {"accuracy" : accuracy_score(y_test, y_pred), 
               "f1" : f1_score(y_test, y_pred)
               ,"auc": roc_auc_score(y_test, y_proba)}
    
    return metrics

def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    n_splits = cfg["modeling"]["validation"]["n_splits"]
    
    combined_path = os.path.join(processed_path, "combined_datasets_with_sentiments.csv")
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    
    #We perform one-hot encoding on the entire dataset once
    #so that every fold has the exact same set of columns
    #(preventing missing columns if a ticker never appears in a specific fold).
    df_encoded = prepare_features(df)
    feature_cols = get_feature_columns(df_encoded)
    
    folds = generate_walk_forward_splits(df_encoded, n_splits= n_splits)
    
    print(f"number of features : {len(feature_cols)}")
    print(f"features : {feature_cols}")
    
    all_metrics = []
    for i, (train_df, test_df) in enumerate(folds):
        metrics = train_and_evalute_folds(train_df, test_df, feature_cols)
        all_metrics.append(metrics)
        
        print(f"Fold : {i+1}")
        print(f"Accuracy : {metrics["accuracy"]}")
        print(f"F1 Score : {metrics["f1"]}")
        print(f"Roc-Auc Score : {metrics['auc']}")
    
    avg_metrics = pd.DataFrame(all_metrics)
    print("\nSummary Between Folds")
    print(avg_metrics.describe().loc[["mean", "std"]])
    

if __name__ == "__main__":
    main()