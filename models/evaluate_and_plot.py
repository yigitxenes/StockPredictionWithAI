import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from models.walk_forward import generate_walk_forward_splits
from models.train_direction_baseline import prepare_features, get_feature_columns
from models.train_volatility_baseline import naive_baseline_predict


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize = (5, 4))
    im = ax.imshow(cm, cmap = "Blues")
    
    labels = ["Declining (0)", "Rising (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True")
    ax.set_title("Direction Model Confusion Matrix \n All Folds Merged")
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha = "center", va = "center",
                    color = "white" if cm[i, j] > cm.max() / 2 else "black", fontsize = 14)
    
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi = 150)
    plt.close()
    print(f"Saved {out_path}")


def plot_actual_vs_predicted(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize = (6, 6))
    ax.scatter(y_true, y_pred, alpha= 0.3, s = 10)
    
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label = "Perfect Guess (y = x)")

    ax.set_xlabel("Real Volatility")
    ax.set_ylabel("Predicted Volatility")
    ax.set_title("Volatility Model Real vs Predicted")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi = 150)
    plt.close()
    
    print(f"Saved {out_path}")
    

def plot_fold_comparison(fold_results, out_path):
    folds = [f"Fold {i + 1}" for i in range(len(fold_results))]
    model_rmse = [r["rmse_model"] for r in fold_results]
    naive_rmse = [r["rmse_naive"] for r in fold_results]
    
    x = np.arange(len(folds))
    width = 0.35
    
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.bar(x - width / 2, model_rmse, width, label="Model (XGBoost)", color="#4C72B0")
    ax.bar(x + width / 2, naive_rmse, width, label="Naive Baseline", color="#DD8452")
    
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title("Volatility Model : Model vs Naive Per Fold")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi = 150)
    plt.close()
    print(f"Saved {out_path}")
    
def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    reports_path = cfg.get("reports_path", "reports")
    os.makedirs(reports_path, exist_ok=True)
    n_splits = cfg["modeling"]["validation"]["n_splits"]

    combined_path = os.path.join(processed_path, "combined_datasets.csv")
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    df_encoded = prepare_features(df)
    feature_cols = get_feature_columns(df_encoded)
    folds = generate_walk_forward_splits(df_encoded, n_splits)

    
    all_y_true_dir, all_y_pred_dir = [], []

    
    all_y_true_vol, all_y_pred_vol = [], []
    fold_results = []

    for i, (train_df, test_df) in enumerate(folds):
        X_train, X_test = train_df[feature_cols], test_df[feature_cols]

        # Direction
        y_train_dir, y_test_dir = train_df["target_direction"], test_df["target_direction"]
        clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             eval_metric="logloss", random_state=42)
        clf.fit(X_train, y_train_dir)
        y_pred_dir = clf.predict(X_test)
        all_y_true_dir.extend(y_test_dir.tolist())
        all_y_pred_dir.extend(y_pred_dir.tolist())

        # Volatility
        y_train_vol, y_test_vol = train_df["target_volatility"], test_df["target_volatility"]
        reg = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        reg.fit(X_train, y_train_vol)
        y_pred_vol = reg.predict(X_test)
        all_y_true_vol.extend(y_test_vol.tolist())
        all_y_pred_vol.extend(y_pred_vol.tolist())

        naive_pred = naive_baseline_predict(train_df, test_df)
        fold_results.append({
            "rmse_model": np.sqrt(mean_squared_error(y_test_vol, y_pred_vol)),
            "rmse_naive": np.sqrt(mean_squared_error(y_test_vol, naive_pred)),
        })

        print(f"Fold {i + 1} processed.")

    plot_confusion_matrix(
        np.array(all_y_true_dir), np.array(all_y_pred_dir),
        os.path.join(reports_path, "confusion_matrix_direction.png"),
    )
    plot_actual_vs_predicted(
        np.array(all_y_true_vol), np.array(all_y_pred_vol),
        os.path.join(reports_path, "actual_vs_predicted_volatility.png"),
    )
    plot_fold_comparison(
        fold_results,
        os.path.join(reports_path, "fold_comparison_volatility.png"),
    )
    

if __name__ == "__main__":
    main()