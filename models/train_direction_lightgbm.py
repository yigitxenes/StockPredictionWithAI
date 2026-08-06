import os 
import sys

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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

def train_and_evaluate_folds(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols]
    y_train = train_df["target_direction"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["target_direction"]
    
    model = LGBMClassifier(
        n_estimators= 200,
        max_depth= 5,
        learning_rate= 0.05,
        random_state= 42,
        verbose = -1 #shuts off lousy logs
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, -1]
    
    return {"accuracy" : accuracy_score(y_test, y_pred),
            "f1" : f1_score(y_test, y_pred),
            "auc" : roc_auc_score(y_test, y_proba)}


def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    n_splits = cfg["modeling"]["validation"]["n_splits"]
    
    combined_path = os.path.join(processed_path, "combined_datasets.csv")
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    
    df_encoded = prepare_features(df)
    feature_cols = get_feature_columns(df_encoded)
    
    folds = generate_walk_forward_splits(df_encoded, n_splits)
    
    all_metrics = []
    
    for i, (train_df, test_df) in enumerate(folds):
        metrics = train_and_evaluate_folds(train_df, test_df, feature_cols)
        all_metrics.append(metrics)
        print(f"Fold {i + 1}: accuracy={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")
    
    avg_metrics = pd.DataFrame(all_metrics)
    print(f"Summary Between Folds - LightGBM")
    print(avg_metrics.describe().loc[["mean", "std"]])
    

if __name__ == "__main__":
    main()