import os
import sys
import json
from datetime import datetime

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from models.train_direction_baseline import prepare_features as prepare_direction_features, \
    get_feature_columns as get_direction_feature_columns
from models.train_volatility_baseline import prepare_features as prepare_volatility_features, \
    get_feature_cols as get_volatility_feature_columns
from models.tuning import tune_hyperparameters, build_model


# Production modeli BILINCLI olarak teknik+sentiment feature setini kullanir.
# Cross-asset feature'lari test edildi (bkz. ilerleme raporu) ama onceden
# belirlenen anlamlilik esigini (+0.02 AUC, p<0.05) gecemedi ve ek dogrulamada
# (n=10 fold) sonuc zayifladi -- bu yuzden production'a dahil edilmedi.
PRODUCTION_DATASET_FILE = "combined_datasets_with_sentiments.csv"

MODEL_OUTPUT_DIR = "models"


def train_final_direction_model(df, cfg):
    df_encoded = prepare_direction_features(df)
    feature_cols = get_direction_feature_columns(df_encoded)

    n_inner_splits = cfg["modeling"]["hyperparameter_search"]["n_inner_splits"]
    n_trials = cfg["modeling"]["hyperparameter_search"]["n_trials"]

    print("Direction modeli icin nihai hiperparametre araniyor (tum veri uzerinde walk-forward)...")
    best_params = tune_hyperparameters(
        df_encoded, feature_cols, "target_direction",
        model_family="xgboost", task="classification",
        n_inner_splits=n_inner_splits, n_trials=n_trials,
    )

    print(f"Secilen parametreler: {best_params}")
    final_model = build_model("xgboost", "classification", {**best_params, "random_state": 42})

    X = df_encoded[feature_cols]
    y = df_encoded["target_direction"]
    final_model.fit(X, y)

    return final_model, feature_cols, best_params


def train_final_volatility_model(df, cfg):
    df_encoded = prepare_volatility_features(df)
    feature_cols = get_volatility_feature_columns(df_encoded)

    n_inner_splits = cfg["modeling"]["hyperparameter_search"]["n_inner_splits"]
    n_trials = cfg["modeling"]["hyperparameter_search"]["n_trials"]

    print("Volatility modeli icin nihai hiperparametre araniyor (tum veri uzerinde walk-forward)...")
    best_params = tune_hyperparameters(
        df_encoded, feature_cols, "target_volatility",
        model_family="xgboost", task="regression",
        n_inner_splits=n_inner_splits, n_trials=n_trials,
    )

    print(f"Secilen parametreler: {best_params}")
    final_model = build_model("xgboost", "regression", {**best_params, "random_state": 42})

    X = df_encoded[feature_cols]
    y = df_encoded["target_volatility"]
    final_model.fit(X, y)

    return final_model, feature_cols, best_params


def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    tickers = cfg["data"]["tickers"]

    dataset_path = os.path.join(processed_path, PRODUCTION_DATASET_FILE)
    df = pd.read_csv(dataset_path, parse_dates=["Date"])
    print(f"Nihai model egitimi icin dataset: {PRODUCTION_DATASET_FILE}, {len(df)} satir\n")

    # --- Direction ---
    direction_model, direction_feature_cols, direction_params = train_final_direction_model(df, cfg)
    direction_bundle = {
        "model": direction_model,
        "feature_cols": direction_feature_cols,
        "tickers": tickers,
        "best_params": direction_params,
        "dataset_file": PRODUCTION_DATASET_FILE,
        "trained_at": datetime.now().isoformat(),
        "trained_on_date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
    }
    direction_out = os.path.join(MODEL_OUTPUT_DIR, "final_direction_model.pkl")
    joblib.dump(direction_bundle, direction_out)
    print(f"\nKaydedildi: {direction_out}")

    # --- Volatility ---
    volatility_model, volatility_feature_cols, volatility_params = train_final_volatility_model(df, cfg)
    volatility_bundle = {
        "model": volatility_model,
        "feature_cols": volatility_feature_cols,
        "tickers": tickers,
        "best_params": volatility_params,
        "dataset_file": PRODUCTION_DATASET_FILE,
        "trained_at": datetime.now().isoformat(),
        "trained_on_date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
    }
    volatility_out = os.path.join(MODEL_OUTPUT_DIR, "final_volatility_model.pkl")
    joblib.dump(volatility_bundle, volatility_out)
    print(f"Kaydedildi: {volatility_out}")

    # Web app / agent raporu icin, modelin GERCEK performansini (nested
    # walk-forward'dan gelen, ONCEDEN olculmus AUC/RMSE) ayri bir JSON'a
    # yaziyoruz -- boylece rapor uretirken "bu modelin tarihsel AUC'si X"
    # diye DOGRU ve GUNCEL bir sayi kullanabiliriz, uydurmayiz.
    performance_summary = {
        "direction": {
            "metric": "AUC",
            "value": 0.5047,  # nested walk-forward tuned sonucundan (teknik+sentiment, XGBoost)
            "interpretation": "sans seviyesine yakin (0.50), guvenilir bir alim/satim sinyali degildir",
            "note": "Sabit hiperparametreli testte istatistiksel olarak anlamli (p=0.0024) bulunan "
                    "sentiment katkisi, nested tuning altinda dogrulanamamistir (p=0.33).",
        },
        "volatility": {
            "metric": "RMSE",
            "value": 0.01310,
            "naive_baseline_rmse": 0.01443,
            "interpretation": "naive baseline'a (son bilinen volatilite) kiyasla ortalama iyilesme "
                               "gozlemlenmis ancak istatistiksel olarak anlamli degildir (p=0.37)",
        },
        "last_updated": datetime.now().isoformat(),
    }
    perf_out = os.path.join(MODEL_OUTPUT_DIR, "model_performance_summary.json")
    with open(perf_out, "w", encoding="utf-8") as f:
        json.dump(performance_summary, f, ensure_ascii=False, indent=2)
    print(f"Kaydedildi: {perf_out}")


if __name__ == "__main__":
    main()